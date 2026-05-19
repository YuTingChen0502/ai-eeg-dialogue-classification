### **CourseCode:**

### **Group:**

**Student ID:**  
**Name:**

### **PART 1: EEG Brain Signal Classification**

For Part 1, please include the following sections:

1. Method  
2. Preprocessing Design  
3. Experimental Results  
4. Analysis / Discussion

#### **Task 1 Requirements**

- compare at least two preprocessing designs  
- explain which design works better and why

#### **Task 2 Requirements**

- explain how you address cross-subject generalization  
- analyze which methods that work in within-subject setting may not transfer well to cross-subject setting  
- explain the relationship between your local validation performance and public leaderboard performance

**Task 1 Optional Bonus**

- make and discuss at least one meaningful attempt beyond preprocessing  
- examples: different model families, feature extraction, class balancing, augmentation, or other justified design changes

**Task 2 Optional Bonus**

- make and discuss at least one meaningful attempt beyond the basic cross-subject pipeline

## 

## **PART 2: Dialogue Continuity Classification**

**You must answer all the questions/requirements listed below.**

### **Task1 Data Balancing** 

You must **answer all the questions/requirements** listed below**.**

For this part, please choose **the same training model** (TF-IDF \+ SVM/ XGBoost/ Bert….) for comparing the two data balancing methods.

* **Code Screenshots:** Include clear code blocks for both implementations with screenshots in your report. And, explain your code.  
* **Observations:** Please document your findings, and explain everything in details, including  
  * The training model you choose to compare the results.   
  * Briefly explain about random over-sampling.  
  * Explain Macro-F1 first, give a comparison of the Macro-F1 scores resulting from those methods, and why this score.  
  * A determination of which method performed better, explain why.  
  * An explanation of the logic behind your chosen "Advanced Method."  
  * The unique mechanism it uses to help the model learn compared to the random method.  
  * **Best Method Identification:** Clearly state your best methods, and the specific parameters applied, explain why it outperformed and what you have tried.  
  * Any others you find interesting/ like to discuss.

### **Task2 Modeling & Enhancements** 

You must **answer all the questions/requirements** listed below.  
Note that you must at least implement one of the advanced models.

* **Code Screenshots:** Include clear code blocks for both implementations with screenshots in your report. And, explain your code.  
* **Observations:** Please document your findings,and explain everything in details,  including:  
  * A comparison of the Macro-F1 scores resulting from those methods, and explain why this score.  
  * Briefly explain what TF-IDF and SVM is.  
  * A determination of which method performed better, explain why..  
  * An explanation of the logic behind your chosen "Advanced Method."  
  * The unique mechanism it uses to help the model learn.  
  * **Best Model Identification:** Clearly state your best model, the methods used, and the specific parameters applied., explain why it outperformed and what you have tried.  
  * Any others you find interesting/ like to discuss.

**Peer Evaluation**

For the peer evaluation table, each group may rate member contributions on a 1-10 scale. You may decide the score based on your own group collaboration, but the table should briefly indicate each member's contribution.

| Member | Student ID | Name | Score | Contribution  |
| :---- | :---- | :---- | :---- | :---- |
| 01 |  |  |  |  |
| 02 |  |  |  |  |
| 03 |  |  |  |  |
| 04 |  |  |  |  |

### **Report Submission Format Requirements**

To ensure your assignment is graded, please follow these formatting rules:

#### **File Format:**

* Submit your code and report to E3 separately.   
* Please note that each group should **upload only one set** of files in total.  
* **Report:** Submit in **.pdf** format.  
  * **Filename:**CLASSNUMBER\_GROUPNUMBER\_HW3\_report.pdf  
  * For examples:515512\_Group01\_HW3\_report.pdf  
  * Please refer to E3 to check if you're from class **515512 or 515513\.**  
  * **Requirement:** Include all members' student IDs and names at the very beginning of the report.  
  * You should only submit **ONE report** for the entire homework.   
* Incorrect file formats will result in **a score of 0** for this homework.  
* Doesn’t include all members' student ids and names at the very beginning for both code and report will result in **a score of 0** for this homework.