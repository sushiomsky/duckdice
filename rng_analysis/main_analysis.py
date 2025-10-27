#!/usr/bin/env python3
"""
Main RNG Analysis Script
Comprehensive analysis of DuckDice RNG using multiple techniques
"""

import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data_loader import BetHistoryLoader
from pattern_analyzer import RNGPatternAnalyzer
from ml_predictor import RNGMLPredictor
from deep_learning_predictor import DeepLearningRNGPredictor
from visualizer import RNGVisualizer


def print_header(text: str):
    """Print section header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def run_statistical_analysis(df):
    """Run statistical pattern analysis"""
    print_header("STATISTICAL PATTERN ANALYSIS")
    
    analyzer = RNGPatternAnalyzer(df)
    
    # Run all analyses
    analyzer.analyze_distribution()
    analyzer.analyze_autocorrelation()
    analyzer.analyze_runs_test()
    analyzer.analyze_sequential_patterns()
    analyzer.analyze_fourier()
    analyzer.analyze_seed_correlation()
    analyzer.analyze_nonce_patterns()
    
    # Print report
    print("\n")
    print(analyzer.generate_report())
    
    return analyzer


def run_ml_analysis(df):
    """Run machine learning analysis"""
    print_header("MACHINE LEARNING ANALYSIS")
    
    predictor = RNGMLPredictor(df)
    
    # Prepare features
    X, y = predictor.prepare_features()
    
    # Train models
    results = predictor.train_models(X, y)
    
    # Classification analysis
    predictor.evaluate_classification(X, y)
    
    # Cross-validation
    predictor.time_series_cross_validation(X, y)
    
    # Print report
    print("\n")
    print(predictor.generate_ml_report(results))
    
    return predictor, results


def run_deep_learning_analysis(df, epochs=30):
    """Run deep learning analysis"""
    print_header("DEEP LEARNING ANALYSIS")
    
    predictor = DeepLearningRNGPredictor(df)
    
    # Train models
    results = predictor.train_deep_models(sequence_length=50, epochs=epochs, batch_size=64)
    
    # Analyze overfitting
    predictor.analyze_overfitting()
    
    # Print report
    print("\n")
    print(predictor.generate_deep_learning_report(results))
    
    return predictor, results


def create_visualizations(df, output_dir="visualizations"):
    """Create all visualizations"""
    print_header("CREATING VISUALIZATIONS")
    
    visualizer = RNGVisualizer(df)
    visualizer.create_dashboard(output_dir)
    
    return visualizer


def generate_final_report(loader, stat_analyzer, ml_results, dl_results):
    """Generate comprehensive final report"""
    print_header("COMPREHENSIVE RNG ANALYSIS REPORT")
    
    stats = loader.get_basic_stats()
    
    print("DATASET OVERVIEW:")
    print(f"  Total Bets: {stats['total_bets']:,}")
    print(f"  Win Rate: {stats['win_rate']*100:.2f}%")
    print(f"  Number Range: {stats['number_min']} - {stats['number_max']}")
    print(f"  Mean: {stats['number_mean']:.2f} (Expected: 5000)")
    print(f"  Std: {stats['number_std']:.2f}")
    print()
    
    print("STATISTICAL TESTS:")
    if 'distribution' in stat_analyzer.results:
        dist = stat_analyzer.results['distribution']
        print(f"  Distribution Test: {'PASS ✅' if dist['ks_p_value'] > 0.05 else 'FAIL ⚠️'}")
        print(f"  KS p-value: {dist['ks_p_value']:.6f}")
    
    if 'autocorrelation' in stat_analyzer.results:
        auto = stat_analyzer.results['autocorrelation']
        print(f"  Autocorrelation: {'PASS ✅' if not auto['significant_lags'] else 'FAIL ⚠️'}")
        if auto['significant_lags']:
            print(f"    Found at lags: {auto['significant_lags'][:5]}")
    
    if 'runs_test' in stat_analyzer.results:
        runs = stat_analyzer.results['runs_test']
        print(f"  Runs Test: {'PASS ✅' if runs['p_value'] > 0.05 else 'FAIL ⚠️'}")
        print(f"  p-value: {runs['p_value']:.6f}")
    print()
    
    print("MACHINE LEARNING RESULTS:")
    best_ml = max(ml_results.items(), key=lambda x: x[1].get('improvement', -float('inf')))
    print(f"  Best Model: {best_ml[0]}")
    print(f"  Improvement: {best_ml[1].get('improvement', 0):.2f}%")
    print(f"  MAE: {best_ml[1].get('mae', 0):.2f}")
    print()
    
    print("DEEP LEARNING RESULTS:")
    best_dl = max(dl_results.items(), key=lambda x: x[1].get('improvement', -float('inf')) if 'error' not in x[1] else -float('inf'))
    if 'error' not in best_dl[1]:
        print(f"  Best Model: {best_dl[0]}")
        print(f"  Improvement: {best_dl[1].get('improvement', 0):.2f}%")
        print(f"  MAE: {best_dl[1].get('mae', 0):.2f}")
    else:
        print(f"  Training failed or incomplete")
    print()
    
    print("="*70)
    print("FINAL CONCLUSIONS:")
    print("="*70)
    print()
    
    # Analyze results
    max_improvement = max(
        best_ml[1].get('improvement', 0),
        best_dl[1].get('improvement', 0) if 'error' not in best_dl[1] else 0
    )
    
    if max_improvement > 10:
        print("⚠️  WARNING: Models show >10% improvement over baseline")
        print()
        print("This could indicate:")
        print("  1. Overfitting to training data (most likely)")
        print("  2. Random fluctuations being interpreted as patterns")
        print("  3. Artifacts from data collection or preprocessing")
        print()
        print("However, this does NOT mean the RNG is exploitable:")
        print("  - Patterns in historical data don't predict future outcomes")
        print("  - Cryptographic RNG is designed to resist such analysis")
        print("  - These 'patterns' will not persist in live betting")
        print()
    elif max_improvement > 5:
        print("⚠️  Models show moderate improvement (5-10%)")
        print()
        print("This is likely due to:")
        print("  - Overfitting")
        print("  - Random variation (expected in any dataset)")
        print("  - Temporal features that won't generalize")
        print()
        print("The RNG still appears cryptographically secure.")
        print()
    else:
        print("✅ Models show minimal improvement (<5%)")
        print()
        print("This is the expected result for a secure RNG:")
        print("  - No exploitable patterns detected")
        print("  - Historical data provides no predictive power")
        print("  - The RNG appears cryptographically sound")
        print()
    
    print("="*70)
    print("TECHNICAL EXPLANATION:")
    print("="*70)
    print()
    print("DuckDice uses a 'Provably Fair' RNG system:")
    print()
    print("1. Server Seed (secret until revealed)")
    print("2. Client Seed (user-controlled)")
    print("3. Nonce (incremental counter)")
    print("4. Hash Function (likely SHA-256)")
    print()
    print("Result = hash(ServerSeed + ClientSeed + Nonce) mod 10000")
    print()
    print("This design ensures:")
    print("  ✅ Unpredictability: Cannot predict future outcomes")
    print("  ✅ Verifiability: Can verify past results")
    print("  ✅ Fairness: Neither party can manipulate outcomes")
    print()
    print("SHA-256 properties:")
    print("  - Avalanche effect: Small input change → completely different output")
    print("  - One-way function: Cannot reverse engineer inputs")
    print("  - Collision resistant: Infeasible to find two inputs with same output")
    print()
    print("Therefore, even with:")
    print("  - Thousands of historical bets")
    print("  - Advanced machine learning")
    print("  - Deep neural networks")
    print("  - Pattern analysis")
    print()
    print("The RNG remains UNPREDICTABLE and SECURE.")
    print()
    
    print("="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print()
    print("1. DO NOT attempt to exploit perceived patterns")
    print("   → They will not persist in real betting")
    print()
    print("2. DO NOT increase bet sizes based on 'predictions'")
    print("   → Each bet is independent")
    print()
    print("3. DO verify your bets using the verification links")
    print("   → This ensures the casino isn't cheating")
    print()
    print("4. DO gamble responsibly")
    print("   → Only bet what you can afford to lose")
    print()
    print("5. UNDERSTAND the mathematics")
    print("   → House edge ensures long-term casino profit")
    print()
    print("="*70)
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Comprehensive RNG Analysis for DuckDice',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-dir', default='../bet_history',
                       help='Directory containing bet history CSV files')
    parser.add_argument('--skip-dl', action='store_true',
                       help='Skip deep learning analysis (faster)')
    parser.add_argument('--dl-epochs', type=int, default=30,
                       help='Number of epochs for deep learning (default: 30)')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--viz-dir', default='visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    try:
        # ASCII Art Header
        print("\n" + "="*70)
        print("""
    ____             __   ____  _          
   / __ \\__  _______/ /__/ __ \\(_)_______  
  / / / / / / / ___/ //_/ / / / / ___/ _ \\ 
 / /_/ / /_/ / /__/ ,< / /_/ / / /__/  __/ 
/_____/\\__,_/\\___/_/|_/_____/_/\\___/\\___/  
                                           
    RNG ANALYSIS & ATTACK ATTEMPT
        """)
        print("="*70)
        print()
        print("⚠️  DISCLAIMER:")
        print("This tool is for EDUCATIONAL PURPOSES ONLY.")
        print("Attempting to exploit casino RNG is likely to fail and may violate terms of service.")
        print("="*70 + "\n")
        
        # Load data
        print_header("LOADING DATA")
        loader = BetHistoryLoader(args.data_dir)
        df = loader.load_all_files()
        df = loader.preprocess_data()
        loader.print_summary()
        
        # Statistical analysis
        stat_analyzer = run_statistical_analysis(df)
        
        # Machine learning analysis
        ml_predictor, ml_results = run_ml_analysis(df)
        
        # Deep learning analysis (optional, slower)
        dl_results = {}
        if not args.skip_dl:
            dl_predictor, dl_results = run_deep_learning_analysis(df, epochs=args.dl_epochs)
        else:
            print_header("SKIPPING DEEP LEARNING ANALYSIS")
        
        # Visualizations (optional)
        if not args.skip_viz:
            create_visualizations(df, args.viz_dir)
        else:
            print_header("SKIPPING VISUALIZATION GENERATION")
        
        # Final report
        generate_final_report(loader, stat_analyzer, ml_results, dl_results)
        
        print("\n✅ Analysis complete!\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
