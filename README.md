# PSG
Constructing Page Semantic Graphs Using LLMs to Facilitate Analyzing and Understanding Mobile Apps
=======
Constructing Page Semantic Graphs Using LLMs to Facilitate Analyzing and Understanding Mobile Apps. Our code consists of the following two main parts.

## Building PSG
The construction of PSG consists of three phases:
* **Page Information Extraction** : We identify all the pages in an Android app, and extract the UI and code information for each page.

* **Page Semantic Summarization** : 
We fuse the extracted UI and code, and then use the chain of thought approach to guide LLMs to generate a page summary. This part of code is in `Building_PSG/LLM`.

* **PSG Construction** 
We use static analysis to find all the inter-component communication (ICC) between all pages.
 The code for statistic analysis and extraction part is packed to a `jar` executable file in `Building_PSG/statistic_analysis/ppg_sa`.

## Utilizing PSG
### Task 1 App Description Generation
In this task, we evaluate how our proposed method can help generate effective descriptions of mobile apps for users to gain a comprehensive understanding of the app’s behaviors. 
Code in `Utilizing/Task1`

### Task 2 UEWare Detection
we evaluate how PSG can be used to help detecting such special type of malware.
Code in `Utilizing/Task2`.

### Task 3 Authentication Backdoor Analysis
In this task, we aim to utilize the rich UI and code semantics captured by PSG to help detect authentication backdoors in real-world apps more efficiently.
Code in `Utilizing/Task3`