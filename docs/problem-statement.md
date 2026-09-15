# Problem Statement

## Background

Pharmacovigilance — the science of detecting, assessing, and preventing adverse drug reactions (ADRs) — is a mandatory activity for drug manufacturers and regulatory authorities worldwide. The FDA's Adverse Event Reporting System (FAERS) receives hundreds of thousands of spontaneous reports per quarter from healthcare professionals, patients, and manufacturers. These reports form the primary early-warning database for post-market drug safety surveillance.

## The Problem

Identifying genuine ADR safety signals from FAERS requires applying disproportionality analysis methods such as the Proportional Reporting Ratio (PRR) — a statistical technique endorsed by the European Medicines Agency (EMA) and the WHO Uppsala Monitoring Centre. In practice, this work is done manually or with expensive specialist software:

- Analysts must construct and evaluate 2×2 contingency tables for every drug–event pair in a dataset.
- Flagged signals must be cross-referenced with reporting trends over time to determine whether a pattern is emerging, stable, or declining.
- Each signal must then be manually traced to the specific ICH M4 Common Technical Document (CTD) sections that are required to document it in a regulatory submission (NDA, BLA, ANDA, or MAA).
- CTD dossier completeness must be assessed across five modules (M1–M5), with critical safety-relevant sections weighted differently from standard sections.

None of these steps are automated in a single, integrated workflow. Pharmacovigilance teams and regulatory affairs professionals maintain separate tools for signal detection, dossier management, and submission readiness — with no automated bridge between them.

## Who Is Affected

- **Drug safety scientists** at pharmaceutical companies who monitor post-market safety data and must detect ADR signals in a timely, auditable way.
- **Regulatory affairs professionals** who prepare CTD dossiers for FDA/EMA submission and need to ensure that every detected safety signal is documented in the correct sections before filing.
- **Smaller biotech and pharmaceutical organisations** that cannot afford enterprise pharmacovigilance platforms and rely on manual spreadsheets and ad-hoc scripts.

## Why It Matters

A missed or delayed ADR signal can lead to patient harm that might have been preventable. Conversely, a submission dossier with undocumented safety findings risks rejection or a request for additional information from the regulatory authority, delaying market authorisation by months or years.

The manual process is slow, inconsistent across analysts, and error-prone — particularly for organisations managing multiple drugs and high report volumes simultaneously.

## Why Existing Solutions Fall Short

Enterprise pharmacovigilance platforms (such as Oracle Argus or Veeva Vault Safety) address signal detection but do not provide an integrated, conversational interface for regulatory traceability — i.e., they do not automatically answer the question: "Given this flagged signal, which CTD sections in my current dossier draft are missing or incomplete?" That mapping step is performed manually by regulatory affairs staff using domain expertise and institutional knowledge.

PharmaGuard AI addresses this gap by combining signal detection, priority ranking, temporal trend analysis, CTD dossier readiness scoring, and signal-to-CTD-section traceability in a single open platform — with an IBM Bob MCP integration that makes the entire workflow accessible through natural-language conversation.
