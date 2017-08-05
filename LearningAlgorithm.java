//package batchProcessing.util;

/**
 * This is interface for j48 and naive bayes simple algorithm.
 * @author Sagar
 *
 */
public interface LearningAlgorithm {
	public void initialize(String s1, String s2);
	public void initialize(String s1, String s2, int m, int n);
	public void process();
	public double getMinRootMeanError();
	public String getBestCaseSummary();
	public void processTest();
}
