#!/usr/bin/env python3
"""
Clair Agent - Command Line Interface
Week 1 Finale - Day 7

Usage:
    python clair.py --daily          # Generate daily report
    python clair.py --thread-only    # Just generate thread
    python clair.py --stats          # Show database stats
"""

import argparse
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description='Clair Agent - AI Research Aggregator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clair.py --daily          Generate full daily report
  python clair.py --thread-only    Just generate a thread
  python clair.py --stats          Show database statistics
  
Week 1 Complete! 🎉
Built over 7 days with zero API costs.
        """
    )
    
    parser.add_argument(
        '--daily',
        action='store_true',
        help='Generate full daily report (papers + thread + report)'
    )
    
    parser.add_argument(
        '--thread-only',
        action='store_true',
        help='Generate thread only (faster)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show ChromaDB statistics'
    )
    
    parser.add_argument(
        '--days-back',
        type=int,
        default=3,
        help='How many days back to search arXiv (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Show banner
    print("=" * 60)
    print("🤖 CLAIR AGENT - Week 1 Edition")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: ", end="")
    
    if args.stats:
        print("Database Statistics\n")
        show_stats()
    elif args.thread_only:
        print("Thread Generation Only\n")
        generate_thread_only(args.days_back)
    elif args.daily:
        print("Full Daily Report\n")
        generate_daily_report(args.days_back)
    else:
        parser.print_help()
        sys.exit(0)


def show_stats():
    """Show ChromaDB statistics"""
    try:
        from chromadb import PersistentClient
        from chromadb.config import Settings
        import config
        
        client = PersistentClient(
            path=config.CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        print("\n📊 CHROMADB STATISTICS\n")
        
        collections = {
            'arxiv_papers': 'arXiv Papers',
            'hackernews_stories': 'Hacker News Stories',
            'huggingface_papers': 'Hugging Face Papers'
        }
        
        total = 0
        for coll_name, display_name in collections.items():
            try:
                coll = client.get_collection(coll_name)
                count = coll.count()
                total += count
                print(f"  {display_name:.<40} {count:>5}")
            except:
                print(f"  {display_name:.<40} {'0':>5}")
        
        print(f"  {'─' * 40}  {'─' * 5}")
        print(f"  {'TOTAL':.<40} {total:>5}\n")
        
        print("✅ Database healthy!")
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")


def generate_thread_only(days_back: int):
    """Generate thread only (quick mode)"""
    print(f"📚 Fetching papers (last {days_back} days)...")
    print("⚠️  Thread-only mode not yet implemented")
    print("    Use --daily for full report")
    print("\n💡 TIP: Run notebook code/day 006.py for now")


def generate_daily_report(days_back: int):
    """Generate full daily report"""
    print(f"\n🔄 Running full daily pipeline...\n")
    print(f"1. Fetching arXiv papers (last {days_back} days)...")
    print("2. Scanning Hacker News...")
    print("3. Checking Hugging Face...")
    print("4. Cross-referencing...")
    print("5. Calculating scores...")
    print("6. Generating thread...")
    print("7. Creating report...\n")
    
    print("⚠️  Full automation not yet implemented")
    print("    This is your Week 2 goal!")
    print("\n💡 FOR NOW:")
    print("    1. Run notebook: code/day 006.py")
    print("    2. It generates everything automatically")
    print("    3. Week 2 will make this CLI work end-to-end")
    print("\n📅 COMING IN WEEK 2:")
    print("    - Full CLI automation")
    print("    - Memory (don't repeat topics)")
    print("    - Reflection loops")
    print("    - Voice tuning")


if __name__ == "__main__":
    main()