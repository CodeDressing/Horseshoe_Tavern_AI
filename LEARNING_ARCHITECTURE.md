# Horseshoe Tavern AI Learning Architecture

## Purpose

The platform stores customer interactions, detects language patterns,
learns spelling and phrasing variants, creates candidate training examples,
and improves its models through a controlled review and evaluation process.

## Core rule

Raw customer messages may improve language understanding, but they may not
directly modify verified business facts such as:

- Hours
- Prices
- Menu items
- Events
- Specials
- Policies
- Contact details
- Private-event terms
- Availability

## Learning workflow

1. Receive customer input.
2. Preserve the original message.
3. Normalize text.
4. Detect possible spelling corrections.
5. Detect intent and entities.
6. Retrieve verified business knowledge.
7. Generate a natural response variant.
8. Validate factual consistency.
9. Store the complete interaction.
10. Collect explicit and implicit feedback.
11. Place uncertain examples into a review queue.
12. Approve, edit, or reject training examples.
13. Build a versioned training dataset.
14. Train a candidate model.
15. Evaluate against production and regression tests.
16. Promote only when the candidate satisfies all gates.
17. Preserve rollback capability.

## Privacy rule

Sensitive data must be minimized, redacted where appropriate, protected by
access controls, and governed by a documented retention policy.
