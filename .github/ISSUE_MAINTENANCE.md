# Issue Maintenance Guidelines

This document provides guidelines for maintaining GitHub issues in the FinGPT repository, including identification and handling of spam/inappropriate content.

## Spam Issue Identification

Spam issues typically exhibit one or more of the following characteristics:

### Common Spam Patterns

1. **Commercial Advertising**
   - Promotional content for commercial services/products
   - External links to payment processing, financial services, or unrelated commercial sites
   - "Free API" offers that require registration on external platforms
   - QR codes or contact information for external services

2. **Blank or Template Issues**
   - Issues that contain only the default GitHub issue template text
   - Empty feature requests with no actual content
   - Issues that don't describe a problem, feature request, or question

3. **Off-Topic Content**
   - Financial guides or articles unrelated to FinGPT development
   - General financial advice or market analysis
   - Content that belongs in blog posts, not issue trackers

### Identified Spam Issues

The following issues have been identified as spam and should be closed by maintainers:

- **#250** - "实战经验：将 FinGPT 情感信号融合进多因子 A 股量化框架 + 免费开放 API 接入"
  - Type: Commercial advertising for external API service
  - Contains promotional content, QR codes, and external service links
  - Action: Close with comment about advertising policy

- **#245** - "saham"
  - Type: Blank feature request
  - Contains only default GitHub issue template text
  - Action: Close requesting actual content

- **#222** - "Top Small Business Payment Processing in the USA"
  - Type: Commercial advertising
  - External links to payment processing services
  - Action: Close as off-topic advertising

- **#219** - "Customer has had to turn off FIN"
  - Type: Commercial advertising
  - Promotional content disguised as a customer story
  - Action: Close as advertising

- **#211** - "Market Linked Debentures: A Simple and Complete Guide"
  - Type: Off-topic financial guide
  - General financial content unrelated to FinGPT development
  - Action: Close as off-topic

## Handling Spam Issues

### For Maintainers with Permissions

When closing spam issues, use the following comment template:

```
This issue appears to be [spam type: advertising/blank/off-topic] and does not relate to the FinGPT open-source project development. 

GitHub Issues should be used for:
- Bug reports with reproduction steps
- Feature requests with clear descriptions
- Legitimate questions about FinGPT usage or development

For [advertising/promotional content], please use GitHub Discussions or consider reaching out through appropriate channels mentioned in our CONTRIBUTING.md.

For [general financial questions], please check our FAQ.md or use GitHub Discussions.

This issue is being closed to maintain the quality and focus of the issue tracker.
```

### For Contributors

If you identify spam issues that maintainers should address:

1. **Comment on the issue** (politely) noting that it appears to be spam
2. **Create a new issue** titled "Issue Maintenance: [Issue Number] appears to be spam"
3. **Reference this document** and explain why the issue should be closed
4. **Include evidence** such as:
   - External commercial links
   - Default template text
   - Off-topic content examples

## Automated Detection

### Manual Detection Script

Use the provided `scripts/identify_spam_issues.py` script to automatically detect potential spam issues:

```bash
python scripts/identify_spam_issues.py
```

This script checks for:
- Default template content
- External commercial links
- Short/blank issue bodies
- Known spam patterns

### GitHub Action Integration

The repository includes a GitHub Action (`.github/workflows/issue-triage.yml`) that automatically:
- Detects spam patterns in new issues
- Labels potential spam issues with the "spam" label
- Adds automated comments with confidence scores and reasoning
- Automatically categorizes legitimate issues (bug, enhancement, documentation, question)

The GitHub Action uses the same spam detection patterns as the manual script and runs automatically when issues are created or edited.

## Prevention

### Issue Templates

Our issue templates (`.github/ISSUE_TEMPLATE/`) help prevent spam by:
- Requiring specific information
- Providing clear guidance on what constitutes a valid issue
- Including links to appropriate alternative channels

### Contributor Guidelines

Direct contributors to:
- Read `CONTRIBUTING.md` before opening issues
- Use GitHub Discussions for general questions
- Check existing issues before creating new ones

## Regular Maintenance

Maintainers should:
1. Review new issues within 24-48 hours
2. Close spam issues promptly with appropriate comments
3. Update this document with new spam patterns as they emerge
4. Run the spam detection script weekly

## Additional Resources

- [GitHub Community Guidelines](https://docs.github.com/en/site-policy/github-terms/github-community-guidelines)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)