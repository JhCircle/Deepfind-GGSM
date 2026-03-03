import os
import json
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from loguru import logger
import sys

from config import GGSMConfig
from utils import get_dataloader
from model import FindGGSM


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def setup_logger(rank):
    """Setup logger - only print from rank 0"""
    if rank != 0:
        logger.remove()
        logger.add(sys.stderr, level="CRITICAL")
    else:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )


def setup_lora(model, config, is_main=True):
    """Setup LoRA for the model"""
    
    llm_model = model.llm_model
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    model.llm_model = get_peft_model(llm_model, lora_config)
    
    if is_main:
        logger.info("=" * 70)
        logger.info("🎯 LoRA Configuration:")
        logger.info(f"  Rank (r): {config.lora_r}")
        logger.info(f"  Alpha: {config.lora_alpha}")
        logger.info(f"  Dropout: {config.lora_dropout}")
        logger.info(f"  Bias: {config.lora_bias}")
        logger.info(f"  Target Modules: {config.lora_target_modules}")
        logger.info("=" * 70)
        
        # Print trainable parameters
        model.llm_model.print_trainable_parameters()
    
    return model


def monitor_gradients(model, writer, global_step, prefix="gradients", is_main=True):
    """Monitor gradient statistics - only log from rank 0"""
    if not is_main:
        return
    
    total_norm = 0.0
    param_count = 0
    grad_norm_sum = 0.0
    max_grad_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            grad_norm_sum += param_norm.item()
            param_count += 1
            max_grad_norm = max(max_grad_norm, param_norm.item())
    
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
        avg_norm = grad_norm_sum / param_count
        
        writer.add_scalar(f"{prefix}/total_norm", total_norm, global_step)
        writer.add_scalar(f"{prefix}/avg_norm", avg_norm, global_step)
        writer.add_scalar(f"{prefix}/max_norm", max_grad_norm, global_step)
        
        logger.info(f"Step {global_step:4d} - {prefix}: "
                   f"Total: {total_norm:8.2f}, "
                   f"Avg: {avg_norm:8.6f}, "
                   f"Max: {max_grad_norm:8.4f}")


def main(rank=0, world_size=1):
    """Main training function"""
    is_main = rank == 0
    
    setup_logger(rank)
    
    # Load config
    config = GGSMConfig()
    config.batch_size = config.real_batch // (config.gradient_accumulation_steps * world_size)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        if world_size > 1:
            setup_distributed(rank, world_size)
    else:
        device = torch.device('cpu')
    
    # Setup logging and TensorBoard
    if is_main:
        logger.info("=" * 70)
        logger.info("🚀 Starting Training with LoRA")
        logger.info("=" * 70)
        logger.info(f"📱 Device: {device}")
        logger.info(f"🌍 World Size: {world_size}")
        logger.info(f"🔢 Rank: {rank}")
        logger.info("=" * 70)
        logger.info("⚙️  Configuration:")
        logger.info(f"  LLM Model: {config.llm_model_name}")
        logger.info(f"  Batch Size: {config.batch_size}")
        logger.info(f"  Real Batch: {config.real_batch}")
        logger.info(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
        logger.info(f"  Learning Rate: {config.lr}")
        logger.info(f"  Warmup Steps: {config.warmup_steps}")
        logger.info(f"  Total Steps: {config.total_training_steps}")
        logger.info(f"  EMA Decay: {config.ema_decay}")
        logger.info("=" * 70)
        
        os.makedirs(config.output_dir, exist_ok=True)
        log_dir = os.path.join(config.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        config_path = os.path.join(config.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        logger.info(f"✅ Saved config to {config_path}")
    else:
        writer = None
    
    # Load tokenizer
    if is_main:
        logger.info(f"📥 Loading tokenizer from {config.llm_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_main:
        logger.info("✅ Tokenizer loaded")
    
    # Load dataset
    if is_main:
        logger.info(f"📂 Loading dataset from alpaca_data_cleaned.json")
    train_dataloader, dataset_size = get_dataloader(
        config, tokenizer, 'alpaca_data_cleaned.json', is_train=True
    )
    if is_main:
        logger.info(f"✅ Dataset loaded: {dataset_size} samples")
        logger.info(f"📊 Number of batches: {len(train_dataloader)}")
    
    # Initialize model
    if is_main:
        logger.info(f"🏗️  Building model")
    model = FindGGSM(config, device).to(device)
    model.llm_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if is_main:
        logger.info("✅ Model built and moved to device")
    
    # Setup LoRA
    model = setup_lora(model, config, is_main=is_main)
    
    # Distributed wrapper
    if world_size > 1:
        if is_main:
            logger.info(f"🔄 Wrapping model with DDP")
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        dist.barrier()
        if is_main:
            logger.info("✅ DDP wrapper applied")
    
    # Setup optimizer - only optimize trainable parameters
    if is_main:
        logger.info(f"⚙️  Setting up optimizer")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
    if is_main:
        logger.info(f"✅ Optimizer created for {len(trainable_params)} trainable parameters")
    
    # Setup learning rate schedulers
    if is_main:
        logger.info(f"📈 Setting up learning rate schedulers")
    total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = config.warmup_ratio * total_steps // config.gradient_accumulation_steps
    start_factor = config.lr / (warmup_steps + 1e-8)
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_steps]
    )
    if is_main:
        logger.info(f"✅ Schedulers created")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info("=" * 70)
    
    # Start training
    model.train()
    global_step = 0
    real_update_steps = 0
    
    for epoch in range(config.num_epochs):
        if world_size > 1:
            dist.barrier()
        
        if is_main:
            logger.info(f"\n🔄 Epoch {epoch + 1}/{config.num_epochs}")
        
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            disable=not is_main,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}"
        )
        
        for step, batch in pbar:
            current_step = global_step
            
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            answer_input_ids = batch['answer_input_ids'].to(device)
            answer_attention_mask = batch['answer_attention_mask'].to(device)
            
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                              dtype=torch.bfloat16):
                loss_contra, avg_weight, mode = model(
                    query_input_ids, query_attention_mask,
                    answer_input_ids, answer_attention_mask,
                    current_step=current_step
                )
            
            if is_main:
                epoch_loss += loss_contra.item()
                num_batches += 1
                
                if step % 10 == 0:
                    logger.info(
                        f"  Step {step:4d} | Loss: {loss_contra:.4f} | "
                        f"Weight: {avg_weight:.6f} | Mode: {mode} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                    )
                    writer.add_scalar("loss/contrastive", loss_contra.item(), global_step=global_step)
                    writer.add_scalar("weight/avg", avg_weight, global_step=global_step)
                    writer.add_scalar("lr/learning_rate", optimizer.param_groups[0]['lr'], global_step=global_step)
                
                pbar.set_postfix({
                    'loss': f'{loss_contra:.4f}',
                    'weight': f'{avg_weight:.6f}'
                })
            
            global_step += 1
            
            loss = loss_contra / config.gradient_accumulation_steps
            loss.backward()
            
            if current_step < config.warmup_steps:
                model_module = model.module if isinstance(model, DDP) else model
                # 注意：这里可能需要调整以适应 LoRA 包装的模型
                if hasattr(model_module, 'estimate_future_importance_from_grads'):
                    model_module.estimate_future_importance_from_grads(
                        query_input_ids, query_attention_mask, current_step=current_step
                    )
            
            if is_main and step % 50 == 0:
                monitor_gradients(model, writer, global_step, "model_gradients", is_main=True)
            
            if config.clip_grad_flag:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
            
            if world_size > 1:
                dist.barrier()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                real_update_steps += 1
                
                if world_size > 1:
                    dist.barrier()
                
                if is_main and real_update_steps % config.save_steps_num == 0:

                    final_save_dir = os.path.join(config.output_dir, "final_model")
                    os.makedirs(final_save_dir, exist_ok=True)
                    
                    model_module = model.module if isinstance(model, DDP) else model
                    
                    lora_final_dir = os.path.join(final_save_dir, "lora_weights")
                    os.makedirs(lora_final_dir, exist_ok=True)
                    model_module.llm_model.save_pretrained(lora_final_dir)  # PEFT标准保存
                    
                    torch.save(
                        model_module.low_dim_projector.state_dict(),
                        os.path.join(final_save_dir, 'projector_weights.pt')
                    )
                    
                    tokenizer.save_pretrained(final_save_dir)
                    with open(os.path.join(final_save_dir, 'training_config.json'), 'w') as f:
                        json.dump(config.__dict__, f, indent=2)
        if is_main:
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"📊 Epoch {epoch + 1} completed - Avg Loss: {avg_epoch_loss:.4f}")
            writer.add_scalar("loss/epoch_avg", avg_epoch_loss, global_step=epoch)
    
    # Training completed
    if is_main:
        logger.info("\n" + "=" * 70)
        logger.info("✅ Training completed!")
        logger.info("=" * 70)
        
        # Save final LoRA model
        final_save_dir = os.path.join(config.output_dir, "final_model")
        os.makedirs(final_save_dir, exist_ok=True)
        
        model_module = model.module if isinstance(model, DDP) else model
        model_module.save_pretrained(final_save_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(final_save_dir)
        
        # Save final config
        with open(os.path.join(final_save_dir, 'config.json'), 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        logger.info(f"✅ Final LoRA model saved to {final_save_dir}")
        writer.close()
    
    # Cleanup distributed
    if world_size > 1:
        cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DeepFind GGSM model with LoRA')
    parser.add_argument('--rank', type=int, default=0, help='Process rank')
    parser.add_argument('--world_size', type=int, default=1, help='Total processes')
    
    args = parser.parse_args()
    
    main(rank=args.rank, world_size=args.world_size)

