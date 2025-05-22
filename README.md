# Guerilla Compression - Demo Repository

## What is Guerilla Compression?

Guerilla Compression is a proof-of-concept library designed to demonstrate a novel approach to compressing time-series data, particularly financial market data (snapshots and tick data), for efficient storage and retrieval, especially in database environments (SQL Server, PostgreSQL) and local file systems (Pickle/Parquet).

The core idea revolves around specialized encoding techniques for timestamp and numerical columns, combined with a chunking mechanism that allows for parallel processing and partial data retrieval based on time ranges or feature lookups.

---

## **⚠️ IMPORTANT NOTICE: DEMONSTRATION PURPOSES ONLY ⚠️**

**This repository contains code strictly for demonstration and presentation purposes. It is NOT intended for production use, evaluation for production, or any other application without the explicit prior written consent of the author.**

*   **Core Algorithm Withheld:** The central, high-performance compression algorithms and certain key optimization techniques that make Guerilla Compression competitive in terms of compression factor and speed **are intentionally withheld from this public repository.** This is because the full library is being developed with plans for future monetization.
*   **Presentation Material:** The code provided here is a simplified and incomplete version, primarily to support a video presentation and showcase the general architecture and API design.
*   **DO NOT USE:** You are expressly prohibited from using, copying, modifying, distributing, or deploying any part of this code for any purpose other than viewing it as a demonstration accompanying its presentation.
*   **Monetization Intent:** The author is actively exploring avenues to monetize the complete Guerilla Compression library. Unauthorized use of this demo code undermines these efforts.

**By accessing or viewing this code, you acknowledge and agree to these terms. If you are interested in the full capabilities of Guerilla Compression or potential licensing, please contact the author.**

---

## Regarding the Tests

The tests included in the `/tests` directory are currently **placeholders and are not fully robust or comprehensive.** They serve to illustrate the basic structure of a test suite and demonstrate that testing is a considered part of the development process.

Due to the demo nature of this repository and the withholding of core components, the tests do not cover all edge cases, performance characteristics, or the full spectrum of functionalities that would be present in a production-ready library. They are provided "as-is" for illustrative purposes only.

---

## Contact

For inquiries regarding the full Guerilla Compression library, potential collaborations, or licensing, please await further announcements or contact channels that will be provided by the author in the future.

*Note: A `pyproject.toml` file is not included at this early proof-of-concept stage. Dependency management for this demo is handled via `requirements.txt`.*
