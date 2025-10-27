#!/usr/bin/env python3
"""
Balance tracking example for DuckDice
Tracks all currency balances and displays them in a formatted table
"""

import subprocess
import json
import sys
from datetime import datetime
from typing import Dict, List, Any


def get_user_info(api_key: str) -> Dict[Any, Any]:
    """Get user information from DuckDice API"""
    cmd = [
        "python3", "duckdice.py",
        "--api-key", api_key,
        "--json",
        "user-info"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    
    return json.loads(result.stdout)


def format_balance(value: str) -> str:
    """Format balance value for display"""
    if value is None:
        return "0.00000000"
    try:
        return f"{float(value):.8f}"
    except:
        return value


def display_balances(user_info: Dict[Any, Any]):
    """Display all balances in a formatted table"""
    print("=" * 80)
    print(f"BALANCE TRACKER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"User: {user_info.get('username', 'N/A')}")
    print(f"Level: {user_info.get('level', 'N/A')} 🐾 (Absolute: {user_info.get('absoluteLevel', {}).get('level', 'N/A')})")
    print("=" * 80)
    
    balances = user_info.get('balances', [])
    
    if not balances:
        print("No balances found.")
        return
    
    # Table header
    print(f"\n{'Currency':<10} {'Main Balance':<20} {'Faucet Balance':<20} {'Affiliate Balance':<20}")
    print("-" * 80)
    
    # Calculate totals (simplified - would need exchange rates for accurate total)
    total_currencies = len(balances)
    
    for balance in balances:
        currency = balance.get('currency', 'N/A')
        main = format_balance(balance.get('main'))
        faucet = format_balance(balance.get('faucet'))
        affiliate = format_balance(balance.get('affiliate'))
        
        print(f"{currency:<10} {main:<20} {faucet:<20} {affiliate:<20}")
    
    print("-" * 80)
    print(f"Total Currencies: {total_currencies}")
    print("=" * 80)


def display_wagered(user_info: Dict[Any, Any]):
    """Display wagered amounts"""
    wagered = user_info.get('wagered', [])
    
    if not wagered:
        print("\nNo wagering data available.")
        return
    
    print("\n" + "=" * 80)
    print("WAGERED AMOUNTS")
    print("=" * 80)
    print(f"{'Currency':<10} {'Amount':<30}")
    print("-" * 80)
    
    for wager in wagered:
        currency = wager.get('currency', 'N/A')
        amount = wager.get('amount', 'N/A')
        print(f"{currency:<10} {amount:<30}")
    
    print("=" * 80)


def display_bonuses(user_info: Dict[Any, Any]):
    """Display active wagering bonuses"""
    bonuses = user_info.get('wageringBonuses', [])
    
    if not bonuses:
        print("\nNo active wagering bonuses.")
        return
    
    print("\n" + "=" * 80)
    print("ACTIVE WAGERING BONUSES")
    print("=" * 80)
    
    for bonus in bonuses:
        print(f"\nBonus: {bonus.get('name', 'N/A')}")
        print(f"Type: {bonus.get('type', 'N/A')}")
        print(f"Hash: {bonus.get('hash', 'N/A')}")
    
    print("=" * 80)


def display_tles(user_info: Dict[Any, Any]):
    """Display Time Limited Events"""
    tles = user_info.get('tle', [])
    
    if not tles:
        print("\nNo Time Limited Events.")
        return
    
    print("\n" + "=" * 80)
    print("TIME LIMITED EVENTS (TLEs)")
    print("=" * 80)
    
    active_tles = [tle for tle in tles if tle.get('status') == 'active']
    finished_tles = [tle for tle in tles if tle.get('status') == 'finished']
    
    if active_tles:
        print("\n🟢 Active TLEs:")
        for tle in active_tles:
            print(f"  - {tle.get('name', 'N/A')} (Hash: {tle.get('hash', 'N/A')})")
    
    if finished_tles:
        print("\n⚫ Finished TLEs:")
        for tle in finished_tles:
            print(f"  - {tle.get('name', 'N/A')} (Hash: {tle.get('hash', 'N/A')})")
    
    print("=" * 80)


def display_affiliate_info(user_info: Dict[Any, Any]):
    """Display affiliate information"""
    campaign = user_info.get('campaign')
    affiliate = user_info.get('affiliate')
    
    if not campaign and not affiliate:
        return
    
    print("\n" + "=" * 80)
    print("AFFILIATE INFORMATION")
    print("=" * 80)
    
    if campaign:
        print(f"Campaign ID: {campaign}")
    if affiliate:
        print(f"Referred by: {affiliate}")
    
    print("=" * 80)


def save_to_file(user_info: Dict[Any, Any], filename: str = "balance_history.json"):
    """Save balance snapshot to file for historical tracking"""
    snapshot = {
        'timestamp': datetime.now().isoformat(),
        'username': user_info.get('username'),
        'level': user_info.get('level'),
        'balances': user_info.get('balances'),
        'wagered': user_info.get('wagered')
    }
    
    try:
        # Load existing history
        try:
            with open(filename, 'r') as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        # Append new snapshot
        history.append(snapshot)
        
        # Save updated history
        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✅ Balance snapshot saved to {filename}")
    except Exception as e:
        print(f"\n❌ Failed to save snapshot: {e}", file=sys.stderr)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python3 balance_tracker.py <API_KEY> [--save]")
        print("       python3 balance_tracker.py <API_KEY> --save balance.json")
        sys.exit(1)
    
    api_key = sys.argv[1]
    save_snapshot = '--save' in sys.argv
    
    # Get user info
    print("Fetching user information...")
    user_info = get_user_info(api_key)
    
    # Display all information
    display_balances(user_info)
    display_wagered(user_info)
    display_bonuses(user_info)
    display_tles(user_info)
    display_affiliate_info(user_info)
    
    # Save snapshot if requested
    if save_snapshot:
        filename = "balance_history.json"
        # Check if custom filename provided
        try:
            save_idx = sys.argv.index('--save')
            if save_idx + 1 < len(sys.argv):
                filename = sys.argv[save_idx + 1]
        except (ValueError, IndexError):
            pass
        
        save_to_file(user_info, filename)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
