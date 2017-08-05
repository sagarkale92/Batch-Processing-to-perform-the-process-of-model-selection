//package batchProcessing.driver;

import weka.classifiers.bayes.NaiveBayesSimple;
import weka.classifiers.trees.J48;

/**
 * This is driver code to classify instances using J48 and Naive Bayes Simple.
 * @author Sagar
 *
 */
public class Driver {
	/**
	 * This is main function with training and testing filenames as arguments.
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			if (args.length < 4) {
				System.out.println("Invalid Arguments");
				System.exit(0);
			}
			System.out.println(
					"Decision pruning, model selection and compare learning algorithms by cross validation.\n");
			LearningAlgorithm tree = new J48Tree();
			LearningAlgorithm nBC = new NaiveBayesSmp();
			tree.initialize(args[0], args[1], Integer.parseInt(args[2]), Integer.parseInt(args[3]));
			tree.process();
			System.out.println("\nFor Training data:");
			System.out.println(tree.getBestCaseSummary());
			nBC.initialize(args[0], args[1]);
			nBC.process();
			System.out.println("\nFor Training data:");
			System.out.println(nBC.getBestCaseSummary());
			System.out.println(
					"------------------------------------------------------------------------------------------------------------------------------------------------");
			
			if (tree.getMinRootMeanError() < nBC.getMinRootMeanError()) {
				System.out.println("Hence, The best model produced on entire training set is J48 Decision Tree.");
				weka.core.SerializationHelper.write("best.model", new J48());
				System.out.println("\nFor Testing data by Best Model:");
				tree.processTest();
			} else {
				System.out.println(
						"Hence, The best model produced on entire training set is Naive Bayes Simple Classifier.");
				weka.core.SerializationHelper.write("best.model", new NaiveBayesSimple());
				System.out.println("\nFor Testing data by Best Model::");
				nBC.processTest();
			}

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}
}