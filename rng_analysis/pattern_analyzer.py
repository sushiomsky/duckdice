#!/usr/bin/env python3
"""
Pattern analysis for DuckDice RNG
Analyzes patterns, correlations, and statistical properties
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RNGPatternAnalyzer:
    """Analyze RNG patterns and statistical properties"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
    
    def analyze_distribution(self) -> Dict:
        """Analyze the distribution of random numbers"""
        print("\n" + "="*60)
        print("DISTRIBUTION ANALYSIS")
        print("="*60)
        
        numbers = self.df['Number'].dropna()
        
        # Chi-square test for uniformity
        observed, _ = np.histogram(numbers, bins=100, range=(0, 10000))
        expected = np.full(100, len(numbers) / 100)
        chi2_stat, chi2_p = stats.chisquare(observed, expected)
        
        # Kolmogorov-Smirnov test for uniform distribution
        ks_stat, ks_p = stats.kstest(numbers / 10000, 'uniform')
        
        # Anderson-Darling test
        # Normalize to 0-1 range
        normalized = (numbers - numbers.min()) / (numbers.max() - numbers.min())
        
        results = {
            'mean': numbers.mean(),
            'median': numbers.median(),
            'std': numbers.std(),
            'expected_mean': 5000,
            'expected_std': np.sqrt((10000**2 - 1) / 12),  # For uniform 0-9999
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_p,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'skewness': stats.skew(numbers),
            'kurtosis': stats.kurtosis(numbers),
        }
        
        print(f"Mean: {results['mean']:.2f} (Expected: {results['expected_mean']})")
        print(f"Median: {results['median']:.2f}")
        print(f"Std: {results['std']:.2f} (Expected: {results['expected_std']:.2f})")
        print(f"Skewness: {results['skewness']:.4f} (0 = symmetric)")
        print(f"Kurtosis: {results['kurtosis']:.4f} (0 = normal)")
        print(f"\nChi-square test: p-value = {chi2_p:.6f}")
        print(f"KS test: p-value = {ks_p:.6f}")
        print(f"{'PASS' if ks_p > 0.05 else 'FAIL'}: Distribution appears {'uniform' if ks_p > 0.05 else 'non-uniform'}")
        
        self.results['distribution'] = results
        return results
    
    def analyze_autocorrelation(self, max_lag: int = 50) -> Dict:
        """Analyze autocorrelation in the sequence"""
        print("\n" + "="*60)
        print("AUTOCORRELATION ANALYSIS")
        print("="*60)
        
        numbers = self.df['Number'].dropna().values
        
        # Calculate autocorrelation
        autocorr = []
        for lag in range(1, max_lag + 1):
            if len(numbers) > lag:
                corr = np.corrcoef(numbers[:-lag], numbers[lag:])[0, 1]
                autocorr.append(corr)
            else:
                autocorr.append(0)
        
        # Find significant correlations (|r| > 2/sqrt(n))
        threshold = 2 / np.sqrt(len(numbers))
        significant_lags = [i+1 for i, r in enumerate(autocorr) if abs(r) > threshold]
        
        results = {
            'autocorrelations': autocorr,
            'max_autocorr': max(autocorr, key=abs),
            'max_lag': autocorr.index(max(autocorr, key=abs)) + 1,
            'significant_lags': significant_lags,
            'threshold': threshold,
        }
        
        print(f"Threshold for significance: ±{threshold:.4f}")
        print(f"Max autocorrelation: {results['max_autocorr']:.4f} at lag {results['max_lag']}")
        
        if significant_lags:
            print(f"⚠️  Significant correlations found at lags: {significant_lags[:10]}")
        else:
            print("✅ No significant autocorrelations detected")
        
        self.results['autocorrelation'] = results
        return results
    
    def analyze_runs_test(self) -> Dict:
        """Runs test for randomness"""
        print("\n" + "="*60)
        print("RUNS TEST")
        print("="*60)
        
        # Test on numbers above/below median
        numbers = self.df['Number'].dropna().values
        median = np.median(numbers)
        binary = (numbers > median).astype(int)
        
        # Count runs
        runs = 1 + np.sum(binary[1:] != binary[:-1])
        n1 = np.sum(binary)
        n2 = len(binary) - n1
        
        # Expected runs and standard deviation
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / 
                          ((n1 + n2)**2 * (n1 + n2 - 1)))
        
        # Z-score
        z_score = (runs - expected_runs) / std_runs
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        results = {
            'runs': runs,
            'expected_runs': expected_runs,
            'z_score': z_score,
            'p_value': p_value,
        }
        
        print(f"Observed runs: {runs}")
        print(f"Expected runs: {expected_runs:.2f}")
        print(f"Z-score: {z_score:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"{'PASS' if p_value > 0.05 else 'FAIL'}: Sequence appears {'random' if p_value > 0.05 else 'non-random'}")
        
        self.results['runs_test'] = results
        return results
    
    def analyze_sequential_patterns(self) -> Dict:
        """Analyze sequential patterns"""
        print("\n" + "="*60)
        print("SEQUENTIAL PATTERN ANALYSIS")
        print("="*60)
        
        numbers = self.df['Number'].dropna().values
        
        # Consecutive differences
        diffs = np.diff(numbers)
        
        # Analyze gap patterns
        gap_stats = {
            'mean_gap': np.mean(np.abs(diffs)),
            'std_gap': np.std(diffs),
            'max_gap': np.max(np.abs(diffs)),
            'positive_gaps': np.sum(diffs > 0) / len(diffs),
        }
        
        # Look for repeating patterns
        repeat_2 = np.sum(numbers[1:] == numbers[:-1])
        repeat_distance = []
        for i in range(len(numbers) - 1):
            try:
                next_idx = np.where(numbers[i+1:] == numbers[i])[0]
                if len(next_idx) > 0:
                    repeat_distance.append(next_idx[0] + 1)
            except:
                pass
        
        results = {
            'gap_stats': gap_stats,
            'consecutive_repeats': repeat_2,
            'avg_repeat_distance': np.mean(repeat_distance) if repeat_distance else None,
        }
        
        print(f"Mean absolute gap: {gap_stats['mean_gap']:.2f}")
        print(f"Std of gaps: {gap_stats['std_gap']:.2f}")
        print(f"Max gap: {gap_stats['max_gap']}")
        print(f"Consecutive repeats: {repeat_2}")
        print(f"% positive changes: {gap_stats['positive_gaps']*100:.2f}%")
        
        self.results['sequential'] = results
        return results
    
    def analyze_fourier(self, max_freq: int = 100) -> Dict:
        """Fourier analysis to detect periodic patterns"""
        print("\n" + "="*60)
        print("FOURIER ANALYSIS (FREQUENCY DOMAIN)")
        print("="*60)
        
        numbers = self.df['Number'].dropna().values
        
        # Remove mean
        numbers_centered = numbers - np.mean(numbers)
        
        # FFT
        N = len(numbers_centered)
        yf = fft(numbers_centered)
        xf = fftfreq(N)[:N//2]
        
        # Power spectrum
        power = 2.0/N * np.abs(yf[:N//2])
        
        # Find dominant frequencies
        top_freq_idx = np.argsort(power)[-10:][::-1]
        top_frequencies = xf[top_freq_idx]
        top_powers = power[top_freq_idx]
        
        results = {
            'frequencies': xf[:max_freq],
            'power': power[:max_freq],
            'top_frequencies': top_frequencies,
            'top_powers': top_powers,
        }
        
        print(f"Top 5 frequencies (cycles per bet):")
        for i in range(min(5, len(top_frequencies))):
            if top_frequencies[i] > 0:
                period = 1 / top_frequencies[i]
                print(f"  {i+1}. Frequency: {top_frequencies[i]:.6f}, Period: {period:.2f} bets, Power: {top_powers[i]:.2f}")
        
        # Check if any frequency is significantly strong
        mean_power = np.mean(power)
        std_power = np.std(power)
        significant = top_powers[0] > mean_power + 3 * std_power
        
        if significant:
            print(f"⚠️  Significant periodic pattern detected!")
        else:
            print("✅ No significant periodic patterns")
        
        self.results['fourier'] = results
        return results
    
    def analyze_seed_correlation(self) -> Dict:
        """Analyze correlation between seeds and outcomes"""
        print("\n" + "="*60)
        print("SEED CORRELATION ANALYSIS")
        print("="*60)
        
        # Group by server seed
        seed_groups = self.df.groupby('ServerSeed').agg({
            'Number': ['mean', 'std', 'count'],
            'Result_Binary': 'mean'
        }).reset_index()
        
        seed_groups.columns = ['ServerSeed', 'Num_Mean', 'Num_Std', 'Count', 'Win_Rate']
        
        results = {
            'unique_seeds': len(seed_groups),
            'avg_bets_per_seed': seed_groups['Count'].mean(),
            'seed_number_variance': seed_groups['Num_Mean'].std(),
            'seed_winrate_variance': seed_groups['Win_Rate'].std(),
        }
        
        print(f"Unique server seeds: {results['unique_seeds']}")
        print(f"Avg bets per seed: {results['avg_bets_per_seed']:.2f}")
        print(f"Variance in mean numbers across seeds: {results['seed_number_variance']:.2f}")
        print(f"Variance in win rates across seeds: {results['seed_winrate_variance']:.4f}")
        
        # Check if certain seeds are "luckier"
        if results['seed_winrate_variance'] > 0.05:
            print("⚠️  High variance in win rates across seeds")
        else:
            print("✅ Win rates consistent across seeds")
        
        self.results['seed_correlation'] = results
        return results
    
    def analyze_nonce_patterns(self) -> Dict:
        """Analyze patterns related to nonce values"""
        print("\n" + "="*60)
        print("NONCE PATTERN ANALYSIS")
        print("="*60)
        
        # Analyze nonce mod patterns
        results = {}
        
        for mod in [10, 100, 1000]:
            nonce_mod = self.df['Nonce'] % mod
            groups = self.df.groupby(nonce_mod).agg({
                'Number': 'mean',
                'Result_Binary': 'mean'
            })
            
            # Check variance
            num_variance = groups['Number'].std()
            win_variance = groups['Result_Binary'].std()
            
            results[f'mod_{mod}'] = {
                'number_variance': num_variance,
                'winrate_variance': win_variance,
            }
            
            print(f"\nNonce mod {mod}:")
            print(f"  Number variance: {num_variance:.2f}")
            print(f"  Win rate variance: {win_variance:.4f}")
        
        self.results['nonce_patterns'] = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("="*60)
        report.append("RNG ANALYSIS REPORT")
        report.append("="*60)
        report.append("")
        
        if 'distribution' in self.results:
            dist = self.results['distribution']
            report.append("1. DISTRIBUTION TEST")
            report.append(f"   Result: {'PASS' if dist['ks_p_value'] > 0.05 else 'FAIL'}")
            report.append(f"   The RNG {'appears uniformly distributed' if dist['ks_p_value'] > 0.05 else 'shows non-uniform distribution'}")
            report.append("")
        
        if 'autocorrelation' in self.results:
            auto = self.results['autocorrelation']
            report.append("2. AUTOCORRELATION TEST")
            report.append(f"   Result: {'PASS' if not auto['significant_lags'] else 'FAIL'}")
            if auto['significant_lags']:
                report.append(f"   Warning: Correlations detected at lags: {auto['significant_lags'][:5]}")
            else:
                report.append("   No significant sequential correlations")
            report.append("")
        
        if 'runs_test' in self.results:
            runs = self.results['runs_test']
            report.append("3. RUNS TEST")
            report.append(f"   Result: {'PASS' if runs['p_value'] > 0.05 else 'FAIL'}")
            report.append(f"   The sequence {'appears random' if runs['p_value'] > 0.05 else 'shows non-random patterns'}")
            report.append("")
        
        report.append("="*60)
        report.append("CONCLUSION")
        report.append("="*60)
        report.append("")
        report.append("⚠️  IMPORTANT: Even if patterns are found, they may be")
        report.append("   due to random chance and NOT exploitable.")
        report.append("")
        report.append("   DuckDice likely uses cryptographic hash functions")
        report.append("   (SHA-256 or similar) which are designed to be")
        report.append("   unpredictable and resistant to pattern analysis.")
        report.append("="*60)
        
        return "\n".join(report)


if __name__ == "__main__":
    from data_loader import BetHistoryLoader
    
    # Load data
    loader = BetHistoryLoader()
    df = loader.load_all_files()
    df = loader.preprocess_data()
    
    # Run analysis
    analyzer = RNGPatternAnalyzer(df)
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
