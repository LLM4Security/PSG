# Static Analysis

### Overview

basicPTG.py performs static analysis on APK files, extracting UI and event listener information, and generates corresponding HTML and JSON files.

### Dependencies

1. Python 3.x
2. Required Python libraries: `multiprocessing`, `json`, `glob`, `subprocess`, `shutil`, etc.
3. Java environment (required to run `ppg_sa.jar` for APK analysis).

### Usage

1. **Modify Paths**:

   Update the following path parameters as needed:

   - `source_path`: Path where the APK files are stored
   - `result_path`: Output path for analysis results

   Example:

   ```python
   source_path = "/the/path/of/apk"
   result_path = "/the/path/of/output"
   ```

2. **Run the Script**:

   Run the following command to execute the script:

   ```bash
   python basicPTG.py
   ```

### Output

- **Static Analysis Results**: The static analysis results for each APK are saved in the specified `result_path`, generating multiple JSON files.
- **HTML and Event Listeners**: The UI HTML and event listener information for each APK are saved as `uiHtml.json` and `uiEvent.json`, respectively.

