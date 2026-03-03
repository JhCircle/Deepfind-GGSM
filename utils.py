import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class TextPairDataset(Dataset):
    """查询-答案对数据集 - 支持 JSON 和 JSONL 格式"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ Data file not found: {data_path}")
        
        logger.info(f"📂 Loading data from {data_path}")
        
        # 根据文件扩展名判断格式
        if data_path.endswith('.json'):
            self._load_json_array(data_path)
        elif data_path.endswith('.jsonl'):
            self._load_jsonl(data_path)
        else:
            # 尝试判断文件格式
            self._load_auto(data_path)
        
        if not self.data:
            raise ValueError(f"❌ No valid data found in {data_path}")
        
        logger.info(f"✅ Loaded {len(self.data)} samples")
    
    def _load_json_array(self, data_path):
        """加载 JSON 数组文件"""
        logger.info("📋 Format detected: JSON Array")
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            
            if not isinstance(data_list, list):
                raise ValueError("JSON file must contain an array")
            
            for line_num, item in enumerate(data_list, 1):
                pair = self._extract_qa_pair(item, line_num)
                if pair:
                    self.data.append(pair)
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            raise
    
    def _load_jsonl(self, data_path):
        """加载 JSONL 文件（每行一个 JSON）"""
        logger.info("📋 Format detected: JSONL")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    pair = self._extract_qa_pair(item, line_num)
                    if pair:
                        self.data.append(pair)
                
                except json.JSONDecodeError as e:
                    logger.debug(f"Line {line_num}: Skipped - {e}")
                    continue
    
    def _load_auto(self, data_path):
        """自动判断格式"""
        logger.info("🔍 Auto-detecting format...")
        
        try:
            # 先尝试作为 JSON 数组
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('['):
                    data_list = json.loads(content)
                    logger.info("✅ Detected as JSON Array")
                    for line_num, item in enumerate(data_list, 1):
                        pair = self._extract_qa_pair(item, line_num)
                        if pair:
                            self.data.append(pair)
                    return
        except:
            pass
        
        # 尝试作为 JSONL
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    pair = self._extract_qa_pair(item, line_num)
                    if pair:
                        self.data.append(pair)
            
            if self.data:
                logger.info("✅ Detected as JSONL")
                return
        except:
            pass
        
        raise ValueError(f"❌ Unable to determine file format for {data_path}")
    
    def _extract_qa_pair(self, item, line_num):
        """从各种格式提取 QA 对"""
        
        query = None
        answer = None
        
        # 格式1: query + answer
        if 'query' in item and 'answer' in item:
            query = str(item['query']).strip()
            answer = str(item['answer']).strip()
        
        # 格式2: instruction + output (Alpaca格式)
        elif 'instruction' in item and 'output' in item:
            instruction = str(item['instruction']).strip()
            input_text = str(item.get('input', '')).strip()
            output = str(item['output']).strip()
            
            if instruction and output:
                query = f"{instruction}\n{input_text}".strip() if input_text else instruction
                answer = output
        
        # 格式3: prompt + completion
        elif 'prompt' in item and 'completion' in item:
            query = str(item['prompt']).strip()
            answer = str(item['completion']).strip()
        
        # 格式4: text (单字段，用 \n\n 分割)
        elif 'text' in item:
            text = str(item['text']).strip()
            parts = text.split('\n\n', 1)
            if len(parts) == 2:
                query, answer = parts[0].strip(), parts[1].strip()
        
        # 验证有效性
        if query and answer:
            return {
                'query': query,
                'answer': answer
            }
        
        return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        answer = item['answer']
        
        # 编码query
        query_encoded = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码answer
        answer_encoded = self.tokenizer(
            answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_encoded['input_ids'].squeeze(0),
            'query_attention_mask': query_encoded['attention_mask'].squeeze(0),
            'answer_input_ids': answer_encoded['input_ids'].squeeze(0),
            'answer_attention_mask': answer_encoded['attention_mask'].squeeze(0),
        }


def collate_fn(batch):
    """批处理函数"""
    return {
        'query_input_ids': torch.stack([item['query_input_ids'] for item in batch]),
        'query_attention_mask': torch.stack([item['query_attention_mask'] for item in batch]),
        'answer_input_ids': torch.stack([item['answer_input_ids'] for item in batch]),
        'answer_attention_mask': torch.stack([item['answer_attention_mask'] for item in batch]),
    }


def get_dataloader(config, tokenizer, data_path, is_train=True):
    """获取数据加载器"""
    dataset = TextPairDataset(data_path, tokenizer, max_length=config.max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_train,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    return dataloader, len(dataset)
