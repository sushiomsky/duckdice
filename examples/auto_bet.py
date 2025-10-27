#!/usr/bin/env python3
"""
Auto-betting example script for DuckDice
This demonstrates how to automate betting with the CLI tool
"""

import subprocess
import json
import time
import sys
from typing import Dict, Any


class AutoBetter:
    """Automated betting system"""
    
    def __init__(self, api_key: str, symbol: str):
        self.api_key = api_key
        self.symbol = symbol
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0
    
    def run_bet(self, amount: str, chance: str, is_high: bool, faucet: bool = False) -> Dict[Any, Any]:
        """Execute a single bet"""
        direction = "--high" if is_high else "--low"
        faucet_flag = ["--faucet"] if faucet else []
        
        cmd = [
            "python3", "duckdice.py",
            "--api-key", self.api_key,
            "--json",
            "dice",
            "--symbol", self.symbol,
            "--amount", amount,
            "--chance", chance,
            direction
        ] + faucet_flag
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        
        return json.loads(result.stdout)
    
    def martingale_strategy(self, base_amount: float, chance: str, max_bets: int = 10):
        """
        Martingale betting strategy: double bet after each loss
        WARNING: This is a high-risk strategy!
        """
        print(f"Starting Martingale Strategy")
        print(f"Base Amount: {base_amount} {self.symbol}")
        print(f"Chance: {chance}%")
        print(f"Max Bets: {max_bets}")
        print("-" * 60)
        
        current_amount = base_amount
        
        for bet_num in range(1, max_bets + 1):
            print(f"\nBet #{bet_num}: {current_amount} {self.symbol}")
            
            response = self.run_bet(str(current_amount), chance, True, faucet=True)
            bet = response['bet']
            
            if bet['result']:
                self.wins += 1
                profit = float(bet['profit'])
                self.total_profit += profit
                print(f"✅ WIN! Profit: {profit} {self.symbol}")
                # Reset to base amount after win
                current_amount = base_amount
            else:
                self.losses += 1
                loss = float(bet['betAmount'])
                self.total_profit -= loss
                print(f"❌ LOSS! Lost: {loss} {self.symbol}")
                # Double the bet after loss
                current_amount *= 2
            
            print(f"Balance: {response['user']['balance']} {self.symbol}")
            print(f"Session Profit: {self.total_profit} {self.symbol}")
            print(f"W/L: {self.wins}/{self.losses}")
            
            time.sleep(1)  # Rate limiting
        
        self.print_summary()
    
    def fixed_strategy(self, amount: float, chance: str, num_bets: int = 10):
        """
        Fixed betting strategy: same amount each time
        This is a safer, more conservative approach
        """
        print(f"Starting Fixed Betting Strategy")
        print(f"Amount: {amount} {self.symbol}")
        print(f"Chance: {chance}%")
        print(f"Number of Bets: {num_bets}")
        print("-" * 60)
        
        for bet_num in range(1, num_bets + 1):
            print(f"\nBet #{bet_num}: {amount} {self.symbol}")
            
            response = self.run_bet(str(amount), chance, True, faucet=True)
            bet = response['bet']
            
            if bet['result']:
                self.wins += 1
                profit = float(bet['profit'])
                self.total_profit += profit
                print(f"✅ WIN! Profit: {profit} {self.symbol}")
            else:
                self.losses += 1
                loss = float(bet['betAmount'])
                self.total_profit -= loss
                print(f"❌ LOSS! Lost: {loss} {self.symbol}")
            
            print(f"Balance: {response['user']['balance']} {self.symbol}")
            print(f"Session Profit: {self.total_profit} {self.symbol}")
            print(f"W/L: {self.wins}/{self.losses}")
            
            time.sleep(1)  # Rate limiting
        
        self.print_summary()
    
    def print_summary(self):
        """Print betting session summary"""
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Bets: {self.wins + self.losses}")
        print(f"Wins: {self.wins}")
        print(f"Losses: {self.losses}")
        print(f"Win Rate: {(self.wins / (self.wins + self.losses) * 100):.2f}%")
        print(f"Total Profit/Loss: {self.total_profit:.8f} {self.symbol}")
        print("=" * 60)


def main():
    """Main entry point"""
    # Configuration
    API_KEY = "your-api-key-here"  # Replace with your API key
    SYMBOL = "XLM"  # Currency to bet with
    
    # Create auto-better instance
    better = AutoBetter(API_KEY, SYMBOL)
    
    # Choose strategy:
    
    # Strategy 1: Fixed betting (safer)
    better.fixed_strategy(
        amount=0.1,
        chance="50",
        num_bets=10
    )
    
    # Strategy 2: Martingale (riskier - commented out by default)
    # better.martingale_strategy(
    #     base_amount=0.01,
    #     chance="50",
    #     max_bets=10
    # )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
