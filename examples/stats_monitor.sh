#!/bin/bash
# Statistics monitoring script
# Displays real-time statistics for multiple currencies

API_KEY="${DUCKDICE_API_KEY:8f9a51ce-af2d-11f0-a08a-524acb1a7d8c}"
CURRENCIES=("BTC" "XLM" "XRP" "ETH" "LTC")
REFRESH_INTERVAL=10  # seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

clear_screen() {
    clear
}

display_header() {
    echo "============================================================"
    echo "           DuckDice Statistics Monitor"
    echo "============================================================"
    echo "Refresh Interval: ${REFRESH_INTERVAL}s"
    echo "Press Ctrl+C to stop"
    echo "============================================================"
    echo ""
}

get_stats() {
    local currency=$1
    python3 ../duckdice.py --api-key "$API_KEY" --json stats --symbol "$currency" 2>/dev/null
}

display_currency_stats() {
    local currency=$1
    local stats=$(get_stats "$currency")
    
    if [ -z "$stats" ]; then
        echo "${RED}[ERROR]${NC} Failed to fetch $currency stats"
        return
    fi
    
    local bets=$(echo "$stats" | jq -r '.bets // "N/A"')
    local wins=$(echo "$stats" | jq -r '.wins // "N/A"')
    local profit=$(echo "$stats" | jq -r '.profit // "N/A"')
    local volume=$(echo "$stats" | jq -r '.volume // "N/A"')
    local main_balance=$(echo "$stats" | jq -r '.balances.main // "N/A"')
    local faucet_balance=$(echo "$stats" | jq -r '.balances.faucet // "N/A"')
    
    # Calculate win rate
    local win_rate="N/A"
    if [ "$bets" != "N/A" ] && [ "$bets" != "0" ]; then
        win_rate=$(awk "BEGIN {printf \"%.2f\", ($wins/$bets)*100}")
    fi
    
    # Color profit based on value
    local profit_color=$NC
    if [ "$profit" != "N/A" ]; then
        if (( $(echo "$profit > 0" | bc -l) )); then
            profit_color=$GREEN
        elif (( $(echo "$profit < 0" | bc -l) )); then
            profit_color=$RED
        fi
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "${YELLOW}$currency${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  Bets: %-15s Wins: %-15s Win Rate: %s%%\n" "$bets" "$wins" "$win_rate"
    printf "  Profit: ${profit_color}%-12s${NC} Volume: %s\n" "$profit" "$volume"
    printf "  Main Balance: %-10s Faucet: %s\n" "$main_balance" "$faucet_balance"
    echo ""
}

monitor_loop() {
    while true; do
        clear_screen
        display_header
        
        for currency in "${CURRENCIES[@]}"; do
            display_currency_stats "$currency"
        done
        
        echo "============================================================"
        echo "Refreshing in ${REFRESH_INTERVAL}s..."
        
        sleep "$REFRESH_INTERVAL"
    done
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed."
    echo "Install it with: sudo apt-get install jq (Ubuntu/Debian)"
    echo "               or: brew install jq (macOS)"
    exit 1
fi

# Check if bc is installed
if ! command -v bc &> /dev/null; then
    echo "Error: bc is required but not installed."
    echo "Install it with: sudo apt-get install bc (Ubuntu/Debian)"
    exit 1
fi

# Start monitoring
monitor_loop
