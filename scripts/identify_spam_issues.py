#!/usr/bin/env python3
"""
Script to identify potential spam issues in the FinGPT repository.
This script uses GitHub CLI to analyze open issues and flag potential spam.
"""

import subprocess
import json
import re
from typing import List, Dict, Set

# Spam detection patterns
SPAM_PATTERNS = {
    'commercial_links': [
        r'greatwestpay\.com',
        r'justpaste\.it',
        r'agentpit\.io',
        r'payment.*processing',
        r'small.*business.*payment',
    ],
    'advertising_keywords': [
        r'free.*api',
        r'completely.*free',
        r'commercial.*service',
        r'payment.*solution',
        r'credit.*card.*processing',
    ],
    'default_template': [
        r'Is your feature request related to a problem\?',
        r'A clear and concise description of what the problem is',
        r'Describe the solution you\'d like',
        r'Describe alternatives you\'ve considered',
        r'Additional context',
    ],
    'off_topic_financial': [
        r'Market Linked Debentures',
        r'payment.*processing.*guide',
        r'small.*business.*payment',
        r'financial.*guide.*comprehensive',
    ],
}

def get_open_issues() -> List[Dict]:
    """Fetch all open issues using GitHub CLI."""
    try:
        result = subprocess.run(
            ['gh', 'issue', 'list', '--repo', 'AI4Finance-Foundation/FinGPT', 
             '--state', 'open', '--json', 'number,title,body,labels,author'],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error fetching issues: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

def analyze_issue_for_spam(issue: Dict) -> Dict:
    """Analyze a single issue for spam indicators."""
    issue_number = issue['number']
    title = issue['title'].lower()
    body = issue.get('body', '').lower()
    full_text = f"{title} {body}"
    
    spam_indicators = {
        'issue_number': issue_number,
        'title': issue['title'],
        'author': issue.get('author', {}).get('login', 'unknown'),
        'spam_type': None,
        'confidence': 0,
        'reasons': [],
    }
    
    # Check for commercial links
    for pattern in SPAM_PATTERNS['commercial_links']:
        if re.search(pattern, full_text, re.IGNORECASE):
            spam_indicators['spam_type'] = 'commercial_advertising'
            spam_indicators['confidence'] += 30
            spam_indicators['reasons'].append(f"Contains commercial link pattern: {pattern}")
    
    # Check for advertising keywords
    for pattern in SPAM_PATTERNS['advertising_keywords']:
        if re.search(pattern, full_text, re.IGNORECASE):
            spam_indicators['spam_type'] = spam_indicators['spam_type'] or 'advertising'
            spam_indicators['confidence'] += 20
            spam_indicators['reasons'].append(f"Contains advertising keyword: {pattern}")
    
    # Check for default template content (blank issues)
    template_matches = 0
    for pattern in SPAM_PATTERNS['default_template']:
        if re.search(pattern, body, re.IGNORECASE):
            template_matches += 1
    
    if template_matches >= 3:  # If most template text is present
        spam_indicators['spam_type'] = spam_indicators['spam_type'] or 'blank_template'
        spam_indicators['confidence'] += 40
        spam_indicators['reasons'].append("Contains default issue template text (likely blank)")
    
    # Check for off-topic financial content
    for pattern in SPAM_PATTERNS['off_topic_financial']:
        if re.search(pattern, full_text, re.IGNORECASE):
            spam_indicators['spam_type'] = spam_indicators['spam_type'] or 'off_topic'
            spam_indicators['confidence'] += 25
            spam_indicators['reasons'].append(f"Contains off-topic financial content: {pattern}")
    
    # Check for very short or empty content
    if len(body.strip()) < 50 and len(body.strip()) > 0:
        spam_indicators['spam_type'] = spam_indicators['spam_type'] or 'low_content'
        spam_indicators['confidence'] += 15
        spam_indicators['reasons'].append("Very short issue body")
    
    # Check for QR code references (common in spam)
    if 'qr code' in full_text or '二维码' in full_text:
        spam_indicators['spam_type'] = spam_indicators['spam_type'] or 'advertising'
        spam_indicators['confidence'] += 25
        spam_indicators['reasons'].append("Contains QR code reference")
    
    return spam_indicators

def main():
    """Main function to identify spam issues."""
    print("🔍 Analyzing open issues for spam patterns...")
    print("=" * 60)
    
    issues = get_open_issues()
    if not issues:
        print("No issues found or error fetching issues.")
        return
    
    print(f"Found {len(issues)} open issues.")
    print()
    
    potential_spam = []
    for issue in issues:
        analysis = analyze_issue_for_spam(issue)
        if analysis['confidence'] >= 30:  # Confidence threshold
            potential_spam.append(analysis)
    
    if not potential_spam:
        print("✅ No potential spam issues detected.")
        return
    
    print(f"⚠️  Found {len(potential_spam)} potential spam issues:")
    print("=" * 60)
    
    # Sort by confidence (highest first)
    potential_spam.sort(key=lambda x: x['confidence'], reverse=True)
    
    for spam in potential_spam:
        print(f"\n🚨 Issue #{spam['issue_number']}: {spam['title']}")
        print(f"   Author: {spam['author']}")
        print(f"   Type: {spam['spam_type']}")
        print(f"   Confidence: {spam['confidence']}%")
        print(f"   Reasons:")
        for reason in spam['reasons']:
            print(f"   - {reason}")
    
    print("\n" + "=" * 60)
    print("📋 Summary:")
    print(f"Total issues analyzed: {len(issues)}")
    print(f"Potential spam issues: {len(potential_spam)}")
    print(f"Legitimate issues: {len(issues) - len(potential_spam)}")
    
    print("\n💡 Recommendations:")
    print("1. Review high-confidence issues (≥70%) for immediate closure")
    print("2. Check medium-confidence issues (30-69%) manually")
    print("3. Update spam patterns in the script if new patterns emerge")
    print("4. Refer to .github/ISSUE_MAINTENANCE.md for handling guidelines")

if __name__ == "__main__":
    main()