#!/usr/bin/env python3
"""
Server Seed Quality Checker
Tests if DuckDice server seeds are cryptographically random or predictable
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from data_loader import BetHistoryLoader


class SeedQualityChecker:
    """Analyze server seed quality and randomness"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        
    def analyze_seed_uniqueness(self) -> Dict:
        """Check how often seeds are reused"""
        print("\n" + "="*60)
        print("SEED UNIQUENESS ANALYSIS")
        print("="*60)
        
        total_bets = len(self.df)
        unique_server_seeds = self.df['ServerSeed'].nunique()
        unique_client_seeds = self.df['ClientSeed'].nunique()
        
        seed_counts = self.df['ServerSeed'].value_counts()
        max_reuse = seed_counts.max()
        avg_reuse = seed_counts.mean()
        
        results = {
            'total_bets': total_bets,
            'unique_server_seeds': unique_server_seeds,
            'unique_client_seeds': unique_client_seeds,
            'max_seed_reuse': max_reuse,
            'avg_seed_reuse': avg_reuse,
            'seed_reuse_distribution': seed_counts.values
        }
        
        print(f"Total Bets: {total_bets:,}")
        print(f"Unique Server Seeds: {unique_server_seeds:,}")
        print(f"Unique Client Seeds: {unique_client_seeds:,}")
        print(f"\nServer Seed Reuse:")
        print(f"  Max uses: {max_reuse:,}")
        print(f"  Average uses: {avg_reuse:.1f}")
        print(f"  Reuse rate: {avg_reuse:.1f} bets per seed")
        
        if max_reuse > 10000:
            print(f"\n⚠️  Some seeds used {max_reuse:,} times!")
            print(f"   This provides LOTS of training data per seed")
        else:
            print(f"\n✅ Seeds rotate frequently (good security)")
        
        self.results['uniqueness'] = results
        return results
    
    def analyze_seed_randomness(self) -> Dict:
        """Test if server seeds are cryptographically random"""
        print("\n" + "="*60)
        print("SEED RANDOMNESS TESTS")
        print("="*60)
        
        unique_seeds = self.df['ServerSeed'].dropna().unique()
        
        if len(unique_seeds) < 10:
            print("⚠️  Too few unique seeds for statistical testing")
            return {}
        
        # Convert hex seeds to integers (use first 16 chars for speed)
        print(f"\nTesting {len(unique_seeds)} unique server seeds...")
        seed_ints = []
        for seed in unique_seeds:
            try:
                seed_int = int(seed[:16], 16)
                seed_ints.append(seed_int)
            except:
                pass
        
        seed_ints = np.array(seed_ints)
        
        # Normalize to 0-1 range for uniformity tests
        normalized = (seed_ints - seed_ints.min()) / (seed_ints.max() - seed_ints.min())
        
        # Test 1: Kolmogorov-Smirnov test for uniformity
        ks_stat, ks_p = stats.kstest(normalized, 'uniform')
        
        # Test 2: Chi-square test
        bins = 20
        observed, _ = np.histogram(normalized, bins=bins)
        expected = np.full(bins, len(normalized) / bins)
        chi2_stat, chi2_p = stats.chisquare(observed, expected)
        
        # Test 3: Runs test
        median = np.median(seed_ints)
        binary = (seed_ints > median).astype(int)
        runs = 1 + np.sum(binary[1:] != binary[:-1])
        n1 = np.sum(binary)
        n2 = len(binary) - n1
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / 
                          ((n1 + n2)**2 * (n1 + n2 - 1)))
        z_score = (runs - expected_runs) / std_runs if std_runs > 0 else 0
        runs_p = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        results = {
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'chi2_statistic': chi2_stat,
            'chi2_p_value': chi2_p,
            'runs_z_score': z_score,
            'runs_p_value': runs_p,
        }
        
        print(f"\n1. Uniformity Test (Kolmogorov-Smirnov):")
        print(f"   Statistic: {ks_stat:.6f}")
        print(f"   P-value: {ks_p:.6f}")
        print(f"   Result: {'PASS ✅' if ks_p > 0.05 else 'FAIL ⚠️'}")
        if ks_p > 0.05:
            print(f"   → Seeds appear uniformly distributed")
        else:
            print(f"   → Seeds may have bias!")
        
        print(f"\n2. Chi-Square Test:")
        print(f"   Statistic: {chi2_stat:.6f}")
        print(f"   P-value: {chi2_p:.6f}")
        print(f"   Result: {'PASS ✅' if chi2_p > 0.05 else 'FAIL ⚠️'}")
        
        print(f"\n3. Runs Test (Randomness):")
        print(f"   Z-score: {z_score:.6f}")
        print(f"   P-value: {runs_p:.6f}")
        print(f"   Result: {'PASS ✅' if runs_p > 0.05 else 'FAIL ⚠️'}")
        
        # Overall assessment
        all_pass = ks_p > 0.05 and chi2_p > 0.05 and runs_p > 0.05
        
        print(f"\n{'='*60}")
        if all_pass:
            print("✅ ALL TESTS PASS")
            print("   Seeds appear cryptographically random")
            print("   Likely using secure PRNG (CSPRNG)")
        else:
            print("⚠️  SOME TESTS FAIL")
            print("   Seeds may have detectable patterns")
            print("   Could indicate weak PRNG")
        print("="*60)
        
        self.results['randomness'] = results
        return results
    
    def analyze_same_seed_sequences(self) -> Dict:
        """Analyze bets with same server seed and client seed"""
        print("\n" + "="*60)
        print("SAME-SEED SEQUENCE ANALYSIS")
        print("="*60)
        
        # Group by both server and client seed
        grouped = self.df.groupby(['ServerSeed', 'ClientSeed'])
        
        sequence_lengths = grouped.size()
        
        results = {
            'total_sequences': len(sequence_lengths),
            'max_sequence_length': sequence_lengths.max(),
            'avg_sequence_length': sequence_lengths.mean(),
            'sequences_over_100': (sequence_lengths >= 100).sum(),
            'sequences_over_1000': (sequence_lengths >= 1000).sum(),
        }
        
        print(f"Total unique (ServerSeed, ClientSeed) pairs: {results['total_sequences']:,}")
        print(f"Max sequence length: {results['max_sequence_length']:,} bets")
        print(f"Average sequence length: {results['avg_sequence_length']:.1f} bets")
        print(f"Sequences with 100+ bets: {results['sequences_over_100']:,}")
        print(f"Sequences with 1000+ bets: {results['sequences_over_1000']:,}")
        
        # Find longest sequence
        longest_idx = sequence_lengths.idxmax()
        longest_server, longest_client = longest_idx
        longest_sequence = self.df[
            (self.df['ServerSeed'] == longest_server) &
            (self.df['ClientSeed'] == longest_client)
        ].sort_values('Nonce')
        
        print(f"\n📊 LONGEST SEQUENCE:")
        print(f"   Server Seed: {longest_server[:32]}...")
        print(f"   Client Seed: {longest_client}")
        print(f"   Length: {len(longest_sequence):,} bets")
        print(f"   Nonce range: {longest_sequence['Nonce'].min()} - {longest_sequence['Nonce'].max()}")
        
        if results['max_sequence_length'] >= 100:
            print(f"\n✅ Found sequences with 100+ bets!")
            print(f"   This is PERFECT for testing ML predictability")
            print(f"   We can train on same (server_seed, client_seed)")
        else:
            print(f"\n⚠️  All sequences are short")
            print(f"   Not enough data per seed to test predictability")
        
        self.results['sequences'] = results
        self.results['longest_sequence'] = longest_sequence
        
        return results
    
    def test_seed_predictability(self) -> Dict:
        """Test if next server seed can be predicted from previous seeds"""
        print("\n" + "="*60)
        print("SEED SEQUENCE PREDICTABILITY TEST")
        print("="*60)
        
        # Get sequence of server seeds in order
        seed_changes = self.df[self.df['ServerSeed'] != self.df['ServerSeed'].shift()]
        unique_seeds_ordered = seed_changes['ServerSeed'].tolist()
        
        if len(unique_seeds_ordered) < 10:
            print("⚠️  Too few seed changes to test predictability")
            return {}
        
        print(f"Testing {len(unique_seeds_ordered)} server seed changes...")
        
        # Convert to integers
        seed_ints = []
        for seed in unique_seeds_ordered:
            try:
                seed_int = int(seed[:16], 16)
                seed_ints.append(seed_int)
            except:
                pass
        
        if len(seed_ints) < 10:
            print("⚠️  Failed to parse seeds")
            return {}
        
        seed_ints = np.array(seed_ints)
        
        # Test autocorrelation
        from scipy.stats import pearsonr
        
        correlations = []
        p_values = []
        
        for lag in range(1, min(10, len(seed_ints) // 2)):
            corr, p_val = pearsonr(seed_ints[:-lag], seed_ints[lag:])
            correlations.append(corr)
            p_values.append(p_val)
        
        max_corr = max(correlations, key=abs)
        max_lag = correlations.index(max_corr) + 1
        
        results = {
            'seed_sequence_length': len(seed_ints),
            'correlations': correlations,
            'p_values': p_values,
            'max_correlation': max_corr,
            'max_correlation_lag': max_lag,
        }
        
        print(f"\nAutocorrelation Analysis:")
        for i, (corr, pval) in enumerate(zip(correlations[:5], p_values[:5]), 1):
            sig = "⚠️ SIGNIFICANT" if abs(corr) > 0.3 or pval < 0.05 else "✅"
            print(f"   Lag {i}: r={corr:.4f}, p={pval:.4f} {sig}")
        
        print(f"\nMax correlation: {max_corr:.4f} at lag {max_lag}")
        
        if abs(max_corr) > 0.3:
            print(f"\n⚠️  WARNING: Significant correlation detected!")
            print(f"   Server seeds may be predictable")
            print(f"   This could indicate weak PRNG")
        else:
            print(f"\n✅ No significant correlations")
            print(f"   Server seeds appear unpredictable")
        
        self.results['seed_predictability'] = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive report"""
        report = []
        report.append("="*60)
        report.append("SEED QUALITY ANALYSIS REPORT")
        report.append("="*60)
        report.append("")
        
        if 'uniqueness' in self.results:
            uniq = self.results['uniqueness']
            report.append("1. SEED UNIQUENESS")
            report.append(f"   Total bets: {uniq['total_bets']:,}")
            report.append(f"   Unique server seeds: {uniq['unique_server_seeds']:,}")
            report.append(f"   Average reuse: {uniq['avg_seed_reuse']:.1f} bets/seed")
            report.append("")
        
        if 'randomness' in self.results:
            rand = self.results['randomness']
            report.append("2. SEED RANDOMNESS")
            ks_pass = rand['ks_p_value'] > 0.05
            chi2_pass = rand['chi2_p_value'] > 0.05
            runs_pass = rand['runs_p_value'] > 0.05
            
            report.append(f"   KS Test: {'PASS ✅' if ks_pass else 'FAIL ⚠️'} (p={rand['ks_p_value']:.4f})")
            report.append(f"   Chi-Square: {'PASS ✅' if chi2_pass else 'FAIL ⚠️'} (p={rand['chi2_p_value']:.4f})")
            report.append(f"   Runs Test: {'PASS ✅' if runs_pass else 'FAIL ⚠️'} (p={rand['runs_p_value']:.4f})")
            report.append("")
        
        if 'sequences' in self.results:
            seq = self.results['sequences']
            report.append("3. SAME-SEED SEQUENCES")
            report.append(f"   Longest sequence: {seq['max_sequence_length']:,} bets")
            report.append(f"   Sequences 100+: {seq['sequences_over_100']:,}")
            report.append(f"   Sequences 1000+: {seq['sequences_over_1000']:,}")
            report.append("")
        
        if 'seed_predictability' in self.results:
            pred = self.results['seed_predictability']
            report.append("4. SEED PREDICTABILITY")
            report.append(f"   Max correlation: {pred['max_correlation']:.4f}")
            if abs(pred['max_correlation']) > 0.3:
                report.append(f"   ⚠️  Seeds may be predictable!")
            else:
                report.append(f"   ✅ Seeds appear unpredictable")
            report.append("")
        
        report.append("="*60)
        report.append("CONCLUSIONS")
        report.append("="*60)
        report.append("")
        
        # Overall assessment
        if 'randomness' in self.results:
            rand = self.results['randomness']
            all_pass = (rand['ks_p_value'] > 0.05 and 
                       rand['chi2_p_value'] > 0.05 and 
                       rand['runs_p_value'] > 0.05)
            
            if all_pass:
                report.append("✅ SERVER SEEDS APPEAR CRYPTOGRAPHICALLY RANDOM")
                report.append("   - All statistical tests pass")
                report.append("   - Likely using secure CSPRNG")
                report.append("   - ML prediction will likely fail")
            else:
                report.append("⚠️  SERVER SEEDS SHOW SOME NON-RANDOMNESS")
                report.append("   - Some statistical tests fail")
                report.append("   - May indicate weak PRNG")
                report.append("   - Worth testing ML predictability")
        
        if 'sequences' in self.results:
            seq = self.results['sequences']
            if seq['sequences_over_100'] > 0:
                report.append("")
                report.append(f"✅ FOUND {seq['sequences_over_100']} SEQUENCES WITH 100+ BETS")
                report.append("   - Good data for testing same-seed predictability")
                report.append("   - Can train ML on same (server_seed, client_seed)")
                report.append("   - Next step: Test if SHA-256 output is predictable")
        
        report.append("")
        report.append("="*60)
        
        return "\n".join(report)


def main():
    """Main entry point"""
    print("="*60)
    print("DUCKDICE SEED QUALITY CHECKER")
    print("="*60)
    print("\nThis tool tests if DuckDice server seeds are:")
    print("  1. Cryptographically random (CSPRNG)")
    print("  2. Predictable from past seeds")
    print("  3. Suitable for ML exploitation attempts")
    print("\n" + "="*60)
    
    # Load data
    print("\nLoading bet history...")
    loader = BetHistoryLoader()
    df = loader.load_all_files()
    df = loader.preprocess_data()
    
    # Create checker
    checker = SeedQualityChecker(df)
    
    # Run analyses
    checker.analyze_seed_uniqueness()
    checker.analyze_seed_randomness()
    checker.analyze_same_seed_sequences()
    checker.test_seed_predictability()
    
    # Print report
    print("\n")
    print(checker.generate_report())
    
    # Save longest sequence for further analysis
    if 'longest_sequence' in checker.results:
        longest = checker.results['longest_sequence']
        output_file = 'longest_same_seed_sequence.csv'
        longest.to_csv(output_file, index=False)
        print(f"\n💾 Saved longest sequence to: {output_file}")
        print(f"   Use this for same-seed ML prediction testing!")


if __name__ == "__main__":
    main()
