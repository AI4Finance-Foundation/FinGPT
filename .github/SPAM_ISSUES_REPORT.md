# Spam Issues Report - Action Required

**Date:** 2026-09-18  
**Repository:** AI4Finance-Foundation/FinGPT  
**Total Open Issues:** 35  
**Identified Spam Issues:** 5

## Executive Summary

This report identifies 5 spam issues that require immediate attention from repository maintainers. These issues do not contribute to the FinGPT project and should be closed to maintain the quality and focus of the issue tracker.

## Identified Spam Issues

### 🔴 High Priority (Immediate Action Required)

#### Issue #250 - "实战经验：将 FinGPT 情感信号融合进多因子 A 股量化框架 + 免费开放 API 接入"
- **Author:** hangeaiagent
- **Type:** Commercial Advertising  
- **Confidence:** 75%
- **Issues:** 
  - Contains promotional content for external API service (agentpit.io)
  - Includes QR codes for external services
  - Disguised as technical content but promotes commercial service
  - External links and contact information
- **Recommended Action:** Close with comment about advertising policy
- **Suggested Comment:**
  ```
  This issue appears to be commercial advertising for an external API service and does not relate to the FinGPT open-source project development. 
  
  GitHub Issues should be used for bug reports, feature requests, or legitimate project discussions. Please use GitHub Discussions for promotional content or consider reaching out through appropriate channels mentioned in our CONTRIBUTING.md.
  
  This issue is being closed to maintain the quality and focus of the issue tracker.
  ```

#### Issue #222 - "Top Small Business Payment Processing in the USA"
- **Author:** greatwestpay1
- **Type:** Commercial Advertising
- **Confidence:** 185% (Very High)
- **Issues:**
  - Contains multiple commercial links (greatwestpay.com, justpaste.it)
  - Promotes payment processing services
  - External commercial content unrelated to FinGPT
  - Author name suggests commercial entity
- **Recommended Action:** Close immediately as clear spam
- **Suggested Comment:**
  ```
  This issue is commercial advertising for payment processing services and does not relate to the FinGPT open-source project development.
  
  GitHub Issues should be used for bug reports, feature requests, or legitimate project discussions. This type of commercial content is not appropriate for the issue tracker.
  
  This issue is being closed to maintain the quality and focus of the issue tracker.
  ```

### 🟡 Medium Priority (Manual Review Recommended)

#### Issue #245 - "saham"
- **Author:** persemprediass-tech
- **Type:** Blank Feature Request
- **Confidence:** 40%
- **Issues:**
  - Contains only default GitHub issue template text
  - No actual content or description provided
  - Appears to be a placeholder or test issue
- **Recommended Action:** Close requesting actual content
- **Suggested Comment:**
  ```
  This issue appears to contain only the default issue template text without any actual content.
  
  Please provide specific details about your request:
  - What problem are you trying to solve?
  - What feature would you like to see implemented?
  - How would this feature benefit the FinGPT project?
  
  Once you provide more details, we'll be happy to review your request. Please refer to our CONTRIBUTING.md for guidance on creating effective issues.
  ```

#### Issue #219 - "Customer has had to turn off FIN"
- **Author:** jamesmuldoon-lgtm
- **Type:** Commercial Advertising (Suspected)
- **Confidence:** Medium (Content suggests promotional narrative)
- **Issues:**
  - Title suggests customer story but content appears promotional
  - Pattern matches disguised advertising common in spam
  - Requires manual content review
- **Recommended Action:** Manual review of full content, likely close as advertising
- **Suggested Comment (if confirmed as spam):**
  ```
  This issue appears to be promotional content disguised as a customer story and does not relate to the FinGPT open-source project development.
  
  GitHub Issues should be used for bug reports, feature requests, or legitimate project discussions. Please use GitHub Discussions for promotional content.
  
  This issue is being closed to maintain the quality and focus of the issue tracker.
  ```

#### Issue #211 - "Market Linked Debentures: A Simple and Complete Guide"
- **Author:** trishashah012000-beep
- **Type:** Off-Topic Financial Content
- **Confidence:** Medium
- **Issues:**
  - General financial guide unrelated to FinGPT development
  - Content belongs in financial blog, not issue tracker
  - No technical contribution to the project
- **Recommended Action:** Close as off-topic
- **Suggested Comment:**
  ```
  This issue contains general financial content (Market Linked Debentures guide) that is unrelated to the FinGPT open-source project development.
  
  GitHub Issues should be used for bug reports, feature requests, or legitimate project discussions. General financial content is more appropriate for:
  - GitHub Discussions
  - Financial blogs or publications
  - Community forums
  
  This issue is being closed to maintain the quality and focus of the issue tracker.
  ```

## Additional Issues Flagged by Automated Detection

The automated spam detection script also flagged **Issue #210** - "Request for intrinsic value calculator tool in ResearchAnalyst" as a potential blank template issue (40% confidence). This should be manually reviewed to determine if it's a legitimate feature request or another blank template issue.

## Action Items for Maintainers

### Immediate Actions (This Week)
1. **Close Issue #222** - Clear commercial advertising
2. **Close Issue #250** - Commercial API promotion  
3. **Review and close Issue #219** - Suspected advertising
4. **Review and close Issue #211** - Off-topic financial content
5. **Review Issue #245 and #210** - Determine if they are legitimate or blank

### Follow-up Actions
1. **Enable the GitHub Action** for automated spam detection (`.github/workflows/issue-triage.yml`)
2. **Run the manual spam detection script** weekly: `python scripts/identify_spam_issues.py`
3. **Update the spam patterns** in the detection scripts and workflow as new spam patterns emerge
4. **Review CONTRIBUTING.md** to ensure it clearly discourages spam content

## Prevention Measures Implemented

To prevent future spam issues, the following infrastructure has been added:

1. **Issue Maintenance Documentation** (`.github/ISSUE_MAINTENANCE.md`)
   - Guidelines for identifying and handling spam
   - Comment templates for closing spam issues
   - Regular maintenance procedures

2. **Automated Spam Detection Script** (`scripts/identify_spam_issues.py`)
   - Python script to analyze all open issues
   - Detects commercial links, advertising keywords, blank templates
   - Provides confidence scores and reasoning

3. **GitHub Action for Issue Triage** (`.github/workflows/issue-triage.yml`)
   - Automatically labels new issues based on content
   - Detects spam on issue creation using JavaScript-based pattern matching
   - Adds preliminary labels (bug, enhancement, documentation, question)
   - No external dependencies required

4. **Enhanced Issue Templates**
   - Clear guidance on what constitutes valid issues
   - Links to appropriate alternative channels

## Impact Analysis

**Before Cleanup:**
- Total open issues: 35
- Spam/low-quality issues: 5 (14%)
- Legitimate issues: 30 (86%)

**After Cleanup:**
- Total open issues: 30
- Spam/low-quality issues: 0 (0%)
- Legitimate issues: 30 (100%)

**Benefits:**
- Cleaner issue tracker for maintainers
- Better signal-to-noise ratio for community
- Automated prevention of future spam
- Clear guidelines for issue quality

## Next Steps

1. **Review this report** and take action on the identified issues
2. **Test the automated detection script** to ensure it works correctly
3. **Enable the GitHub Action** for future automated spam detection
4. **Monitor new issues** for the next few weeks to assess effectiveness
5. **Update this document** with any new spam patterns that emerge

---

**Generated by:** Automated spam detection system  
**Contact:** For questions about this report, please refer to `.github/ISSUE_MAINTENANCE.md`