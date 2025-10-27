#!/usr/bin/env python3
"""
Data loader for DuckDice bet history
Loads and preprocesses CSV files for RNG analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
import re


class BetHistoryLoader:
    """Load and preprocess bet history data"""
    
    def __init__(self, data_dir: str = "../bet_history"):
        self.data_dir = Path(data_dir)
        self.df = None
        
    def load_all_files(self) -> pd.DataFrame:
        """Load all CSV files from bet_history directory"""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    dfs.append(df)
                    print(f"Loaded {len(df)} rows from {file.name}")
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        
        if not dfs:
            raise ValueError("No valid data loaded")
        
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal rows loaded: {len(self.df)}")
        
        return self.df
    
    def extract_seeds_from_url(self, url: str) -> Tuple[str, str]:
        """Extract server seed and client seed from verification URL"""
        server_seed = None
        client_seed = None
        
        if pd.isna(url):
            return None, None
        
        # Extract serverSeed
        server_match = re.search(r'serverSeed=([a-f0-9]+)', url)
        if server_match:
            server_seed = server_match.group(1)
        
        # Extract clientSeed
        client_match = re.search(r'clientSeed=([a-zA-Z0-9]+)', url)
        if client_match:
            client_seed = client_match.group(1)
        
        return server_seed, client_seed
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the data for analysis"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_all_files() first.")
        
        print("\n" + "="*60)
        print("PREPROCESSING DATA")
        print("="*60)
        
        # Convert date to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d %H:%M:%S UTC')
        
        # Convert Result to binary
        self.df['Result_Binary'] = (self.df['Result'] == 'Win').astype(int)
        
        # Extract seeds from verification URL
        print("Extracting seeds from verification URLs...")
        seed_data = self.df['Verification link'].apply(self.extract_seeds_from_url)
        self.df['ServerSeed'] = [s[0] for s in seed_data]
        self.df['ClientSeed'] = [s[1] for s in seed_data]
        
        # Sort by date and nonce
        self.df = self.df.sort_values(['Date', 'Nonce']).reset_index(drop=True)
        
        # Create sequence features
        self.df['Prev_Number'] = self.df['Number'].shift(1)
        self.df['Prev_Result'] = self.df['Result_Binary'].shift(1)
        self.df['Prev_2_Number'] = self.df['Number'].shift(2)
        self.df['Prev_3_Number'] = self.df['Number'].shift(3)
        
        # Rolling statistics
        self.df['Number_Rolling_Mean_10'] = self.df['Number'].rolling(window=10, min_periods=1).mean()
        self.df['Number_Rolling_Std_10'] = self.df['Number'].rolling(window=10, min_periods=1).std()
        self.df['Win_Rate_Last_10'] = self.df['Result_Binary'].rolling(window=10, min_periods=1).mean()
        self.df['Win_Rate_Last_50'] = self.df['Result_Binary'].rolling(window=50, min_periods=1).mean()
        self.df['Win_Rate_Last_100'] = self.df['Result_Binary'].rolling(window=100, min_periods=1).mean()
        
        # Streak features
        self.df['Win_Streak'] = self._calculate_streak(self.df['Result_Binary'])
        self.df['Loss_Streak'] = self._calculate_streak(1 - self.df['Result_Binary'])
        
        # Number patterns
        self.df['Number_Diff'] = self.df['Number'].diff()
        self.df['Number_High'] = (self.df['Number'] > 5000).astype(int)
        self.df['Number_Quartile'] = pd.qcut(self.df['Number'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
        
        # Time-based features
        self.df['Hour'] = self.df['Date'].dt.hour
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['TimeOfDay'] = self.df['Hour'] // 6  # 0=night, 1=morning, 2=afternoon, 3=evening
        
        # Nonce patterns
        self.df['Nonce_Mod_10'] = self.df['Nonce'] % 10
        self.df['Nonce_Mod_100'] = self.df['Nonce'] % 100
        self.df['Nonce_Mod_1000'] = self.df['Nonce'] % 1000
        
        print(f"Preprocessed data shape: {self.df.shape}")
        print(f"Features created: {len(self.df.columns)} columns")
        
        return self.df
    
    def _calculate_streak(self, series: pd.Series) -> pd.Series:
        """Calculate current streak length"""
        streak = series * (series.groupby((series != series.shift()).cumsum()).cumcount() + 1)
        return streak
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for ML models"""
        exclude_cols = ['Date', 'Hash', 'Auto session hash', 'Game name', 
                       'Currency', 'Result', 'Choice', 'Choice Option',
                       'Bet amount', 'Win amount', 'Commission', 'Net profit',
                       'Mining', 'Verification link', 'ServerSeed', 'ClientSeed',
                       'Payout', 'Number']  # Number is the target
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        return feature_cols
    
    def get_basic_stats(self) -> dict:
        """Get basic statistics about the dataset"""
        if self.df is None:
            raise ValueError("No data loaded")
        
        stats = {
            'total_bets': len(self.df),
            'total_wins': self.df['Result_Binary'].sum(),
            'total_losses': len(self.df) - self.df['Result_Binary'].sum(),
            'win_rate': self.df['Result_Binary'].mean(),
            'unique_server_seeds': self.df['ServerSeed'].nunique(),
            'unique_client_seeds': self.df['ClientSeed'].nunique(),
            'number_mean': self.df['Number'].mean(),
            'number_std': self.df['Number'].std(),
            'number_min': self.df['Number'].min(),
            'number_max': self.df['Number'].max(),
            'nonce_range': (self.df['Nonce'].min(), self.df['Nonce'].max()),
            'date_range': (self.df['Date'].min(), self.df['Date'].max()),
        }
        
        return stats
    
    def print_summary(self):
        """Print summary of loaded data"""
        stats = self.get_basic_stats()
        
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"Total Bets: {stats['total_bets']:,}")
        print(f"Wins: {stats['total_wins']:,} ({stats['win_rate']*100:.2f}%)")
        print(f"Losses: {stats['total_losses']:,} ({(1-stats['win_rate'])*100:.2f}%)")
        print(f"\nNumber Statistics:")
        print(f"  Mean: {stats['number_mean']:.2f}")
        print(f"  Std: {stats['number_std']:.2f}")
        print(f"  Range: {stats['number_min']} - {stats['number_max']}")
        print(f"\nSeeds:")
        print(f"  Unique Server Seeds: {stats['unique_server_seeds']}")
        print(f"  Unique Client Seeds: {stats['unique_client_seeds']}")
        print(f"\nNonce Range: {stats['nonce_range'][0]} - {stats['nonce_range'][1]}")
        print(f"Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print("="*60)


if __name__ == "__main__":
    # Test the loader
    loader = BetHistoryLoader()
    df = loader.load_all_files()
    df = loader.preprocess_data()
    loader.print_summary()
    
    print(f"\nFirst few rows:")
    print(df[['Date', 'Nonce', 'Number', 'Result', 'ServerSeed', 'ClientSeed']].head())
    
    print(f"\nFeature columns ({len(loader.get_feature_columns())}):")
    print(loader.get_feature_columns())
