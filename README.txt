CS580L Introduction to Machine Learning
FALL 2016
ASSIGNMENT <ASSIGNMENT 3> README FILE

Due Date: <ASSIGNMENT DUE DATE,  Thursday, November 17, 2016>
Submission Date: <DATE SUBMITED,  Thursday, November 17, 2016>
Author(s): <Sagar Kale>
e-mail(s): <skale4@binghamton.edu>

PURPOSE:
Implement an experiment (batch processing) program which automatically performs the process of model selection.

PERCENT COMPLETE:
100%.

FILES:
Driver.java
J48Tree.java
LearningAlgorithm.java
NaiveBayesSmp.java

SAMPLE OUTPUT:
J48
------------------------------------------------------------------------------------------------------------------------------------------------
+----------+----------+--------------------------------+-----------------------------------+----------------------+----------------------------+
| MinNumObj|  numFolds|  Correctly Classified Instances|   Incorrectly Classified Instances|   Mean absolute error|   Total Number of Instances|
+----------+----------+--------------------------------+-----------------------------------+----------------------+----------------------------+
|         1|         2|                   5524 96.9123%|                        176 3.0877%|                0.0454|                        5700|
|         1|         3|                   5535 97.1053%|                        165 2.8947%|                0.0476|                        5700|
|         1|         4|                   5539 97.1754%|                        161 2.8246%|                0.0444|                        5700|
|         1|         5|                   5532 97.0526%|                        168 2.9474%|                0.0455|                        5700|
|         2|         2|                   5519 96.8246%|                        181 3.1754%|                0.0475|                        5700|
|         2|         3|                   5529 97.0000%|                        171 3.0000%|                0.0488|                        5700|
|         2|         4|                   5535 97.1053%|                        165 2.8947%|                0.0455|                        5700|
|         2|         5|                   5527 96.9649%|                        173 3.0351%|                0.0465|                        5700|
|         3|         2|                   5521 96.8596%|                        179 3.1404%|                0.0490|                        5700|
|         3|         3|                   5527 96.9649%|                        173 3.0351%|                0.0499|                        5700|
|         3|         4|                   5530 97.0175%|                        170 2.9825%|                0.0473|                        5700|
|         3|         5|                   5524 96.9123%|                        176 3.0877%|                0.0482|                        5700|
|         4|         2|                   5521 96.8596%|                        179 3.1404%|                0.0509|                        5700|
|         4|         3|                   5523 96.8947%|                        177 3.1053%|                0.0511|                        5700|
|         4|         4|                   5516 96.7719%|                        184 3.2281%|                0.0496|                        5700|
|         4|         5|                   5519 96.8246%|                        181 3.1754%|                0.0501|                        5700|
+----------+----------+--------------------------------+-----------------------------------+----------------------+----------------------------+


TO COMPILE:
javac -classpath <path for weka jar file> Drivers.java

Ex.
javac -classpath ".:/import/linux/home/skale4/ml/kale_sagar_assignment3/batchProcessing/weka.jar" J48Tree.java
javac -classpath ".:/import/linux/home/skale4/ml/kale_sagar_assignment3/batchProcessing/weka.jar" NaiveBayesSmp.java
javac LearningAlgorithm.java
javac -classpath ".:/import/linux/home/skale4/ml/kale_sagar_assignment3/batchProcessing/weka.jar" Driver.java

TO RUN:
include path for weka and then Driver with training and test files

java -cp <path for weka jar file> Driver trainfile testfile maximumNumberofMinNumObj maximumNumberofNumFolds

Ex.
java -cp ".:/import/linux/home/skale4/ml/kale_sagar_assignment3/batchProcessing/weka.jar" Driver training.arff testing.arff 5 6

TO CLEAN:
rm -f *.model *.class