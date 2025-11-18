import ast
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

class ChessDataset(Dataset):
    def __init__(self, df, target_col):
        # Encode target as class indices
        self.player_to_idx = {name: idx for idx, name in enumerate(df[target_col].unique())}
        self.labels = torch.tensor(df[target_col].map(self.player_to_idx).values, dtype=torch.long)

        # Build deterministic move-to-index mapping
        # TODO Change this to a smart encoding

        all_moves = set()
        for m in df['list_of_moves']:
            moves_list = ast.literal_eval(m)
            all_moves.update(moves_list)
        self.move_to_idx = {move: idx for idx, move in enumerate(sorted(all_moves))}


        # Encode list_of_moves to sequences of integers using the deterministic mapping.        
        self.moves = [
            torch.tensor(self.encode_moves(m), dtype=torch.long)
            for m in df['list_of_moves']
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.moves[idx], self.labels[idx]

    def encode_moves(self, moves):
        moves_list = ast.literal_eval(moves)  # convert string back to list
        return [self.move_to_idx[m] for m in moves_list]  # deterministic integer encoding


def make_datasets(csv_path, target='black_name', splits=(0.8, 0.1, 0.1), seed=42):
    """Load dataset from dataframe, convert player names to integers,
       use deterministic mapping to encode moves to integers, split into
       train, validation and test using predefined split ratio.

       Input: csv_path, target either white or black, split ratios
       
       Output: train_set, val_set, test_set, player indices, number of different players
               torch format for train val and test set"""


    df = pd.read_csv(csv_path)

    dataset = ChessDataset(df, target_col=target)

    num_classes = len(dataset.player_to_idx)

    # Split the dataset
    total_size = len(dataset)
    lengths = [int(total_size * s) for s in splits]
    lengths[-1] = total_size - sum(lengths[:-1])  # fix rounding

    torch.manual_seed(seed)  # for reproducibility
    train_set, val_set, test_set = random_split(dataset, lengths)

    return train_set, val_set, test_set, dataset.player_to_idx, num_classes





