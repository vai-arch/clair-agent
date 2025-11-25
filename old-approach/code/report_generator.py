"""
Clair Agent - Daily Report Generator
Generates beautiful markdown reports for daily AI trends
"""

from datetime import datetime
from typing import List, Dict
import os


class DailyReportGenerator:
    """Generate formatted daily reports"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(
        self,
        papers: List[Dict],
        best_paper: Dict,
        cross_refs: Dict,
        hn_stories: List[Dict],
        hf_papers: List[Dict],
        thread: str,
        generation_time: float
    ) -> str:
        """
        Generate comprehensive daily report
        
        Returns: filepath of generated report
        """
        
        date_str = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f"daily_report_{timestamp}.md")
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# 🤖 Clair Daily Report\n")
            f.write(f"**Date:** {date_str}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%H:%M:%S')}\n\n")
            
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## 📊 Executive Summary\n\n")
            f.write(f"- **Papers Analyzed:** {len(papers)}\n")
            f.write(f"- **HN Stories Scanned:** {len(hn_stories)}\n")
            f.write(f"- **HF Papers Featured:** {len(hf_papers)}\n")
            f.write(f"- **Cross-References Found:** {len(cross_refs['arxiv_hn'])} HN, {len(cross_refs['arxiv_hf'])} HF\n")
            f.write(f"- **Triple Hits:** {len(cross_refs['triple_hits'])} 🏆\n\n")
            
            # Top Paper of the Day
            f.write("---\n\n")
            f.write("## 🏆 Top Paper of the Day\n\n")
            f.write(f"### {best_paper['title']}\n\n")
            f.write(f"**Authors:** {best_paper['authors']}\n\n")
            f.write(f"**Published:** {best_paper['published']}\n\n")
            f.write(f"**arXiv:** [{best_paper['url']}]({best_paper['url']})\n\n")
            
            # Quality Metrics
            f.write("#### Quality Metrics\n\n")
            
            # Signal strength with visual indicator
            signal = best_paper.get('signal_strength', best_paper.get('confidence', 0))
            signal_bar = self._create_progress_bar(signal, 100)
            f.write(f"**Signal Strength:** {signal:.0f}/100 {signal_bar}\n\n")
            
            conf = best_paper.get('confidence', 0)
            conf_bar = self._create_progress_bar(conf, 100)
            f.write(f"**Confidence:** {conf:.0f}% {conf_bar}\n\n")
            
            if 'credibility' in best_paper:
                cred = best_paper['credibility']
                cred_bar = self._create_progress_bar(cred, 100)
                f.write(f"**Source Credibility:** {cred:.0f}/100 {cred_bar}\n\n")
            
            viral = best_paper.get('virality', 0)
            if viral > 0:
                viral_bar = self._create_progress_bar(viral, 100)
                f.write(f"**Virality:** {viral:.0f}/100 {viral_bar}\n\n")
            
            f.write(f"**Platforms:** {best_paper['platforms']}/3 ")
            if best_paper.get('triple_hit'):
                f.write("🏆 *Triple Hit!*")
            f.write("\n\n")
            
            # Summary (first 500 chars)
            f.write("#### Summary\n\n")
            f.write(f"{best_paper['summary'][:500]}...\n\n")
            
            # Generated Thread
            f.write("---\n\n")
            f.write("## 🐦 Generated Thread\n\n")
            f.write("```\n")
            f.write(thread)
            f.write("\n```\n\n")
            
            # Top 5 Papers
            f.write("---\n\n")
            f.write("## 📚 All Papers Ranked\n\n")
            
            for i, paper in enumerate(papers[:5], 1):
                scores = paper.get('scores', {})
                signal = scores.get('signal_strength', scores.get('confidence', 0))
                
                f.write(f"### {i}. {paper['title']}\n\n")
                f.write(f"**Signal:** {signal:.0f} | ")
                f.write(f"**Platforms:** {paper.get('platforms', 1)}/3 | ")
                f.write(f"**Published:** {paper['published'].strftime('%Y-%m-%d')}\n\n")
                f.write(f"[arXiv Link]({paper['url']})\n\n")
                
                # Show engagement if exists
                engagement = paper.get('engagement', {})
                if engagement.get('hn_points', 0) > 0:
                    f.write(f"- 🔥 HN: {engagement['hn_points']} points, {engagement['hn_comments']} comments\n")
                if engagement.get('hf_upvotes', 0) > 0:
                    f.write(f"- 🤗 HF: {engagement['hf_upvotes']} upvotes\n")
                
                f.write("\n")
            
            # Data Sources Summary
            f.write("---\n\n")
            f.write("## 📡 Data Sources\n\n")
            
            f.write("### Hacker News (Top Stories)\n\n")
            for i, story in enumerate(hn_stories[:3], 1):
                f.write(f"{i}. [{story['score']:3d}↑] [{story['title']}]({story['hn_url']})\n")
            f.write(f"\n*...and {len(hn_stories) - 3} more*\n\n")
            
            if hf_papers:
                f.write("### Hugging Face Featured\n\n")
                for i, paper in enumerate(hf_papers[:3], 1):
                    f.write(f"{i}. [{paper['upvotes']:2d}🤗] {paper['title']}\n")
                f.write(f"\n*...and {len(hf_papers) - 3} more*\n\n")
            
            # Footer
            f.write("---\n\n")
            f.write("## 🤖 About This Report\n\n")
            f.write(f"**Generated by:** Clair Agent v0.1 (Week 1)\n\n")
            f.write(f"**Stack:** Ollama + LangChain + ChromaDB + Multi-Source RAG\n\n")
            f.write(f"**Sources:** arXiv + Hacker News + Hugging Face\n\n")
            f.write(f"**Generation Time:** {generation_time:.1f}s\n\n")
            f.write(f"**Cost:** $0.00 (100% local)\n\n")
            f.write("**Methodology:**\n")
            f.write("1. Fetch papers from arXiv (last 3 days)\n")
            f.write("2. Scan HN + HF for cross-references\n")
            f.write("3. Rank by signal strength (confidence × credibility)\n")
            f.write("4. Generate thread with local LLM\n")
            f.write("5. Select best match via semantic search\n\n")
            
            f.write(f"*Built in public over 7 days. Follow the journey on X.*\n")
        
        return filename
    
    def _create_progress_bar(self, value: float, max_value: float, length: int = 20) -> str:
        """Create a visual progress bar"""
        filled = int((value / max_value) * length)
        bar = '█' * filled + '░' * (length - filled)
        
        # Add emoji based on value
        if value >= 70:
            emoji = "💎"
        elif value >= 50:
            emoji = "⭐"
        elif value >= 30:
            emoji = "✓"
        else:
            emoji = "○"
        
        return f"{bar} {emoji}"


# Example usage
if __name__ == "__main__":
    # Test the report generator
    generator = DailyReportGenerator()
    
    test_paper = {
        'title': 'Test Paper on AI',
        'authors': 'Author 1, Author 2',
        'published': datetime.now(),
        'url': 'https://arxiv.org/abs/2024.12345',
        'summary': 'This is a test summary.' * 20,
        'confidence': 75,
        'signal_strength': 70,
        'credibility': 95,
        'virality': 45,
        'platforms': 2,
        'triple_hit': False
    }
    
    print("Test report generator created successfully!")