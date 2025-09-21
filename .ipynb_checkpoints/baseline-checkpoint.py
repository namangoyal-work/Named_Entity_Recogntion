{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "14f0ded6-a90e-427d-9c2e-cb087febacd5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "import torch\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "\n",
    "import transformers\n",
    "\n",
    "from tqdm.notebook import tqdm\n",
    "\n",
    "import os\n",
    "os.environ['TOKENIZERS_PARALLELISM'] = 'true'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "59b36cf5-51f3-4b4a-b9f9-5dce241bdbdd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch.multiprocessing as mp\n",
    "mp.set_start_method('fork', force=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0c621f65-84f9-42a0-b26e-60e605d67197",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "PARAMETERS\n",
    "\"\"\"\n",
    "\n",
    "lr = 1e-3\n",
    "\n",
    "device = torch.device(\"cpu\")\n",
    "if torch.backends.mps.is_available():\n",
    "    device = torch.device(\"mps\")\n",
    "elif torch.cuda.is_available():\n",
    "    device = torch.device(\"cuda\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ef25f389-d19c-4e02-9ece-5c627ab5c2ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "device(type='mps')"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "device"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9ef6e3e5-8518-4c07-9a23-712505ffdb7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# this is ok, need to think of an encoding scheme now...\n",
    "class NERDataset(Dataset):\n",
    "    \n",
    "    def __init__(self, filename, label2id):\n",
    "        \n",
    "        self.sentences = []\n",
    "        with open(filename, 'r') as f:\n",
    "            sentence = []\n",
    "            for l in f:\n",
    "                if l == '\\n':\n",
    "                    self.sentences.append(zip(*sentence))\n",
    "                    sentence = []\n",
    "                else:\n",
    "                    token, cls = l.strip().split('\\t')\n",
    "                    sentence.append((token, label2id[cls]))\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.sentences)\n",
    "        \n",
    "    def __getitem__(self, idx):\n",
    "        return self.sentences[idx]\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "175becd5-9948-4c49-a337-852e408158d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def dl_collate_fn(data):\n",
    "    return tuple(zip(*data))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a3d97d7f-dedf-40b0-9ddd-de45a44ffe7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# token alignment\n",
    "def tokenize_and_align_labels(tokenizer, batch):\n",
    "\n",
    "    tokenized_inputs = tokenizer(batch[0], padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')\n",
    "    labels = []\n",
    "\n",
    "    for i, label in enumerate(batch[1]):\n",
    "        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.\n",
    "        previous_word_idx = None\n",
    "        label_ids = []\n",
    "\n",
    "        for word_idx in word_ids:  # Set the special tokens to -100.\n",
    "            if word_idx is None:\n",
    "                label_ids.append(-100)\n",
    "            elif word_idx != previous_word_idx:  # Only label the first token of a given word.\n",
    "                label_ids.append(label[word_idx])\n",
    "            else:\n",
    "                label_ids.append(-100)\n",
    "\n",
    "            previous_word_idx = word_idx\n",
    "\n",
    "        labels.append(label_ids)\n",
    "\n",
    "    tokenized_inputs[\"labels\"] = torch.tensor(labels, device=device)\n",
    "\n",
    "    return tokenized_inputs"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cf1eeb9a-2e44-4e16-a35d-41357a67d944",
   "metadata": {},
   "source": [
    "## Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c24d2ee1-18c6-425c-a5ec-538cdc20d83b",
   "metadata": {},
   "outputs": [],
   "source": [
    "label2id = {'O': 0,\n",
    " 'B-Biological_Molecule': 1,\n",
    " 'E-Biological_Molecule': 2,\n",
    " 'S-Biological_Molecule': 3,\n",
    " 'I-Biological_Molecule': 4,\n",
    " 'S-Species': 5,\n",
    " 'B-Species': 6,\n",
    " 'I-Species': 7,\n",
    " 'E-Species': 8,\n",
    " 'B-Chemical_Compound': 9,\n",
    " 'E-Chemical_Compound': 10,\n",
    " 'S-Chemical_Compound': 11,\n",
    " 'I-Chemical_Compound': 12}\n",
    "\n",
    "id2label = {v: k for k, v in label2id.items()}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "51b70730-9e83-4769-8c23-da53f0090dae",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_ds = NERDataset('train.txt', label2id)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9c3dfa81-fbb6-421d-8e0e-843ba7f9f8c4",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "val_ds = NERDataset('dev.txt', label2id)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "d04d8134-c2e1-4f4c-825f-812f1a2b9dd7",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dl = DataLoader(train_ds, batch_size=8, collate_fn=dl_collate_fn, num_workers=12, multiprocessing_context=\"fork\", shuffle=True)\n",
    "val_dl = DataLoader(val_ds, batch_size=8, collate_fn=dl_collate_fn, num_workers=12, multiprocessing_context=\"fork\", shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "a04cf71c-7207-425d-bbf7-598bf5d411cd",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of the model checkpoint at distilbert-base-cased were not used when initializing DistilBertForTokenClassification: ['vocab_transform.bias', 'vocab_layer_norm.bias', 'vocab_projector.bias', 'vocab_projector.weight', 'vocab_transform.weight', 'vocab_layer_norm.weight']\n",
      "- This IS expected if you are initializing DistilBertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n",
      "- This IS NOT expected if you are initializing DistilBertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).\n",
      "Some weights of DistilBertForTokenClassification were not initialized from the model checkpoint at distilbert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    }
   ],
   "source": [
    "tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-cased')\n",
    "model = transformers.AutoModelForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=13, id2label=id2label, label2id=label2id)\n",
    "model = model.to(device)\n",
    "optimizer = optim.Adam(model.parameters(), lr=2e-5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "5c615806-088b-453a-b04e-2397199e50c7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4801"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(train_dl)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "f3405d20-e25f-4b25-8eca-c5efde99df77",
   "metadata": {},
   "outputs": [],
   "source": [
    "# train\n",
    "\n",
    "def train():\n",
    "    for epoch in range(10):\n",
    "\n",
    "        print(f'Epoch {epoch+1}:')\n",
    "        train_loss = 0\n",
    "        model.train()\n",
    "        for batch in tqdm(train_dl):\n",
    "            tok_batch = tokenize_and_align_labels(tokenizer, batch)\n",
    "            tok_batch = {k : v.to(device) for k, v in tok_batch.items()}\n",
    "\n",
    "            optimizer.zero_grad()\n",
    "            loss = model(**tok_batch).loss\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "\n",
    "            train_loss += loss.detach()\n",
    "\n",
    "        train_loss = train_loss.cpu()\n",
    "        train_loss /= len(train_dl)\n",
    "        print(f' Train Loss: {train_loss}')\n",
    "\n",
    "        val_loss = 0\n",
    "        model.eval()\n",
    "        for batch in tqdm(val_dl):\n",
    "            tok_batch = tokenize_and_align_labels(tokenizer, batch)\n",
    "            tok_batch = {k : v.to(device) for k, v in tok_batch.items()}\n",
    "\n",
    "            loss = model(**tok_batch).loss\n",
    "            val_loss += loss.detach()\n",
    "\n",
    "        val_loss = val_loss.cpu()\n",
    "        val_loss /= len(val_dl)\n",
    "        print(f' Val Loss: {val_loss}')\n",
    "        print('')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "72d3054c-25ce-4433-a339-98819ffccd92",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1:\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7b865a79479444ce9b49978825295a41",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/4801 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[29], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[43mtrain\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "Cell \u001b[0;32mIn[28], line 16\u001b[0m, in \u001b[0;36mtrain\u001b[0;34m()\u001b[0m\n\u001b[1;32m     14\u001b[0m     loss \u001b[38;5;241m=\u001b[39m model(\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mtok_batch)\u001b[38;5;241m.\u001b[39mloss\n\u001b[1;32m     15\u001b[0m     loss\u001b[38;5;241m.\u001b[39mbackward()\n\u001b[0;32m---> 16\u001b[0m     \u001b[43moptimizer\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mstep\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     18\u001b[0m     train_loss \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m loss\u001b[38;5;241m.\u001b[39mdetach()\n\u001b[1;32m     20\u001b[0m train_loss \u001b[38;5;241m=\u001b[39m train_loss\u001b[38;5;241m.\u001b[39mcpu()\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniforge/base/envs/ml/lib/python3.9/site-packages/torch/optim/optimizer.py:140\u001b[0m, in \u001b[0;36mOptimizer._hook_for_profile.<locals>.profile_hook_step.<locals>.wrapper\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    138\u001b[0m profile_name \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mOptimizer.step#\u001b[39m\u001b[38;5;132;01m{}\u001b[39;00m\u001b[38;5;124m.step\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;241m.\u001b[39mformat(obj\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__class__\u001b[39m\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m)\n\u001b[1;32m    139\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m torch\u001b[38;5;241m.\u001b[39mautograd\u001b[38;5;241m.\u001b[39mprofiler\u001b[38;5;241m.\u001b[39mrecord_function(profile_name):\n\u001b[0;32m--> 140\u001b[0m     out \u001b[38;5;241m=\u001b[39m \u001b[43mfunc\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    141\u001b[0m     obj\u001b[38;5;241m.\u001b[39m_optimizer_step_code()\n\u001b[1;32m    142\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m out\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniforge/base/envs/ml/lib/python3.9/site-packages/torch/optim/optimizer.py:23\u001b[0m, in \u001b[0;36m_use_grad_for_differentiable.<locals>._use_grad\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m     21\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m     22\u001b[0m     torch\u001b[38;5;241m.\u001b[39mset_grad_enabled(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdefaults[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mdifferentiable\u001b[39m\u001b[38;5;124m'\u001b[39m])\n\u001b[0;32m---> 23\u001b[0m     ret \u001b[38;5;241m=\u001b[39m \u001b[43mfunc\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     24\u001b[0m \u001b[38;5;28;01mfinally\u001b[39;00m:\n\u001b[1;32m     25\u001b[0m     torch\u001b[38;5;241m.\u001b[39mset_grad_enabled(prev_grad)\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniforge/base/envs/ml/lib/python3.9/site-packages/torch/optim/adam.py:234\u001b[0m, in \u001b[0;36mAdam.step\u001b[0;34m(self, closure, grad_scaler)\u001b[0m\n\u001b[1;32m    231\u001b[0m                 \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mRuntimeError\u001b[39;00m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m`requires_grad` is not supported for `step` in differentiable mode\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m    232\u001b[0m             state_steps\u001b[38;5;241m.\u001b[39mappend(state[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mstep\u001b[39m\u001b[38;5;124m'\u001b[39m])\n\u001b[0;32m--> 234\u001b[0m     \u001b[43madam\u001b[49m\u001b[43m(\u001b[49m\u001b[43mparams_with_grad\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    235\u001b[0m \u001b[43m         \u001b[49m\u001b[43mgrads\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    236\u001b[0m \u001b[43m         \u001b[49m\u001b[43mexp_avgs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    237\u001b[0m \u001b[43m         \u001b[49m\u001b[43mexp_avg_sqs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    238\u001b[0m \u001b[43m         \u001b[49m\u001b[43mmax_exp_avg_sqs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    239\u001b[0m \u001b[43m         \u001b[49m\u001b[43mstate_steps\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    240\u001b[0m \u001b[43m         \u001b[49m\u001b[43mamsgrad\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgroup\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mamsgrad\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    241\u001b[0m \u001b[43m         \u001b[49m\u001b[43mbeta1\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mbeta1\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    242\u001b[0m \u001b[43m         \u001b[49m\u001b[43mbeta2\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mbeta2\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    243\u001b[0m \u001b[43m         \u001b[49m\u001b[43mlr\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgroup\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mlr\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    244\u001b[0m \u001b[43m         \u001b[49m\u001b[43mweight_decay\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgroup\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mweight_decay\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    245\u001b[0m \u001b[43m         \u001b[49m\u001b[43meps\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgroup\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43meps\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    246\u001b[0m \u001b[43m         \u001b[49m\u001b[43mmaximize\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgroup\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mmaximize\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    247\u001b[0m \u001b[43m         \u001b[49m\u001b[43mforeach\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgroup\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mforeach\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    248\u001b[0m \u001b[43m         \u001b[49m\u001b[43mcapturable\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgroup\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mcapturable\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    249\u001b[0m \u001b[43m         \u001b[49m\u001b[43mdifferentiable\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgroup\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mdifferentiable\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    250\u001b[0m \u001b[43m         \u001b[49m\u001b[43mfused\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgroup\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mfused\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m]\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    251\u001b[0m \u001b[43m         \u001b[49m\u001b[43mgrad_scale\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgrad_scale\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    252\u001b[0m \u001b[43m         \u001b[49m\u001b[43mfound_inf\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mfound_inf\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    254\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m loss\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniforge/base/envs/ml/lib/python3.9/site-packages/torch/optim/adam.py:300\u001b[0m, in \u001b[0;36madam\u001b[0;34m(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, foreach, capturable, differentiable, fused, grad_scale, found_inf, amsgrad, beta1, beta2, lr, weight_decay, eps, maximize)\u001b[0m\n\u001b[1;32m    297\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m    298\u001b[0m     func \u001b[38;5;241m=\u001b[39m _single_tensor_adam\n\u001b[0;32m--> 300\u001b[0m \u001b[43mfunc\u001b[49m\u001b[43m(\u001b[49m\u001b[43mparams\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    301\u001b[0m \u001b[43m     \u001b[49m\u001b[43mgrads\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    302\u001b[0m \u001b[43m     \u001b[49m\u001b[43mexp_avgs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    303\u001b[0m \u001b[43m     \u001b[49m\u001b[43mexp_avg_sqs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    304\u001b[0m \u001b[43m     \u001b[49m\u001b[43mmax_exp_avg_sqs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    305\u001b[0m \u001b[43m     \u001b[49m\u001b[43mstate_steps\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    306\u001b[0m \u001b[43m     \u001b[49m\u001b[43mamsgrad\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mamsgrad\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    307\u001b[0m \u001b[43m     \u001b[49m\u001b[43mbeta1\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mbeta1\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    308\u001b[0m \u001b[43m     \u001b[49m\u001b[43mbeta2\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mbeta2\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    309\u001b[0m \u001b[43m     \u001b[49m\u001b[43mlr\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mlr\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    310\u001b[0m \u001b[43m     \u001b[49m\u001b[43mweight_decay\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mweight_decay\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    311\u001b[0m \u001b[43m     \u001b[49m\u001b[43meps\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43meps\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    312\u001b[0m \u001b[43m     \u001b[49m\u001b[43mmaximize\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mmaximize\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    313\u001b[0m \u001b[43m     \u001b[49m\u001b[43mcapturable\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcapturable\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    314\u001b[0m \u001b[43m     \u001b[49m\u001b[43mdifferentiable\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdifferentiable\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    315\u001b[0m \u001b[43m     \u001b[49m\u001b[43mgrad_scale\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mgrad_scale\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    316\u001b[0m \u001b[43m     \u001b[49m\u001b[43mfound_inf\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mfound_inf\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/opt/homebrew/Caskroom/miniforge/base/envs/ml/lib/python3.9/site-packages/torch/optim/adam.py:363\u001b[0m, in \u001b[0;36m_single_tensor_adam\u001b[0;34m(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf, amsgrad, beta1, beta2, lr, weight_decay, eps, maximize, capturable, differentiable)\u001b[0m\n\u001b[1;32m    360\u001b[0m     param \u001b[38;5;241m=\u001b[39m torch\u001b[38;5;241m.\u001b[39mview_as_real(param)\n\u001b[1;32m    362\u001b[0m \u001b[38;5;66;03m# Decay the first and second moment running average coefficient\u001b[39;00m\n\u001b[0;32m--> 363\u001b[0m \u001b[43mexp_avg\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mmul_\u001b[49m\u001b[43m(\u001b[49m\u001b[43mbeta1\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241m.\u001b[39madd_(grad, alpha\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m \u001b[38;5;241m-\u001b[39m beta1)\n\u001b[1;32m    364\u001b[0m exp_avg_sq\u001b[38;5;241m.\u001b[39mmul_(beta2)\u001b[38;5;241m.\u001b[39maddcmul_(grad, grad\u001b[38;5;241m.\u001b[39mconj(), value\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1\u001b[39m \u001b[38;5;241m-\u001b[39m beta2)\n\u001b[1;32m    366\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m capturable \u001b[38;5;129;01mor\u001b[39;00m differentiable:\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "f50e5843-3628-48d5-9ae9-1b8faeffc048",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n"
     ]
    }
   ],
   "source": [
    "# eval\n",
    "def eval():\n",
    "    for batch in train_dl:\n",
    "        tok_batch = tokenize_and_align_labels(tokenizer, batch)\n",
    "        tok_batch = {k : v.to(device) for k, v in tok_batch.items()}\n",
    "\n",
    "        outputs = model(**tok_batch)\n",
    "        break\n",
    "        \n",
    "if __name__ == \"__main__\":\n",
    "    eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "7029df56-0124-4bfc-8845-0d7dd002e1e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n",
      "[W ParallelNative.cpp:229] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)\n"
     ]
    }
   ],
   "source": [
    "for batch in train_dl:\n",
    "    output = batch\n",
    "    break"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02cc41e2-45a6-4bd6-a5ed-276c04727d44",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4336c8ee-0a7b-4a01-8fef-1e0a5ce824d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "outputs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d0cb943c-c8c2-4a17-81e4-c2a921060a28",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
