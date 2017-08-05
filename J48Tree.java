//package batchProcessing.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class J48Tree implements LearningAlgorithm {
	private String trainfileName;
	private String testfileName;
	private J48 tree = null;
	private BufferedReader trainreader = null;
	private BufferedReader testreader = null;
	private Instances trainingData = null;
	private Instances testingData = null;
	private Evaluation eval = null;
	private double minRootMeanError;
	private int maxM;
	private int maxN;
	private String bestCaseSummary;
	private int[] bestSetting;

	/**
	 * Initializes the parameter.
	 */
	@Override
	public void initialize(String trainfileNameIn, String testfileNameIn, int maxMIn, int maxNIn) {
		try {
			this.setMinRootMeanError(Double.MAX_VALUE);
			maxM = maxMIn;
			maxN = maxNIn;
			setBestSetting(new int[2]);
			System.out.println(
					"J48\n------------------------------------------------------------------------------------------------------------------------------------------------");
			trainfileName = trainfileNameIn;
			testfileName = testfileNameIn;
			trainreader = new BufferedReader(new FileReader(trainfileName));
			trainingData = new Instances(trainreader);
			trainreader.close();
			testreader = new BufferedReader(new FileReader(testfileName));
			testingData = new Instances(testreader);
			testreader.close();
			trainingData.setClassIndex(0);
			testingData.setClassIndex(0);
			setTree(new J48());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(-1);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	@Override
	public void initialize(String s1, String s2) {
	}

	/**
	 * Processes the data using j48 algorithm.
	 */
	@Override
	public void process() {
		System.out.format(
				"+----------+----------+--------------------------------+-----------------------------------+----------------------+----------------------------+%n");
		System.out.format("|%10s|%10s|%32s|%35s|%22s|%28s|%n", "MinNumObj", "numFolds",
				"Correctly Classified Instances", "Incorrectly Classified Instances", "Mean absolute error",
				"Total Number of Instances");
		System.out.format(
				"+----------+----------+--------------------------------+-----------------------------------+----------------------+----------------------------+%n");
		try {
			for (int i = 1; i < maxM; i++) {
				for (int j = 2; j < maxN; j++) {
					process1(i, j);
				}
			}
			System.out.format(
					"+----------+----------+--------------------------------+-----------------------------------+----------------------+----------------------------+%n");
			System.out.println("The best parameter setting is when Minimum number of instances, MinNumObj = "
					+ this.getBestSetting()[0] + " and number of folds, numFolds = " + this.getBestSetting()[1] + ".");
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	/**
	 * processes the data using m and n value.
	 * @param m
	 * @param n
	 */
	public void process1(int m, int n) {
		try {
			tree.setReducedErrorPruning(true);
			tree.setUnpruned(false);
			tree.setMinNumObj(m);
			tree.setNumFolds(n);
			tree.buildClassifier(trainingData);
			Evaluation eval = new Evaluation(trainingData);
			Random rand = new Random(1);
			int folds = 10;
			eval.crossValidateModel(tree, trainingData, folds, rand);
			if (eval.meanAbsoluteError() < this.getMinRootMeanError()) {
				int[] mnbest = { m, n };
				this.setBestSetting(mnbest);
				this.setMinRootMeanError(eval.meanAbsoluteError());
				this.setBestCaseSummary(eval.toSummaryString());
			}
			System.out.format("|%10s|%10s|%32s|%35s|%22s|%28s|%n", m, n,
					(int) eval.correct() + " " + String.format("%.4f", 100 - eval.errorRate() * 100) + "%",
					(int) eval.incorrect() + " " + String.format("%.4f", eval.errorRate() * 100) + "%",
					String.format("%.4f", eval.meanAbsoluteError()), (int) eval.numInstances());
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	/**
	 * Process the test data and generates output.
	 */
	@Override
	public void processTest() {
		try{
			setTree(new J48());
			tree.setReducedErrorPruning(true);
			tree.setUnpruned(false);
			tree.setMinNumObj(getBestSetting()[0]);
			tree.setNumFolds(getBestSetting()[1]);
			tree.buildClassifier(testingData); // build classifier
			Evaluation eval = new Evaluation(testingData);
			eval.evaluateModel(tree, testingData);
			System.out.println(eval.toSummaryString());
			System.out.println("The classification of each test instance by the best model");
            for (int i = 0; i < testingData.numInstances(); i++) {
				double pred = tree.classifyInstance(testingData.instance(i));
				System.out.print("actual: " + testingData.classAttribute().value((int) testingData.instance(i).classValue()));
				System.out.println(", predicted: " + testingData.classAttribute().value((int) pred));
			}
            System.out.println();
			System.out.println("The test error, error: "+String.format("%.4f",((double)eval.incorrect()/eval.numInstances())));
		}
		catch(Exception e){
			e.printStackTrace();
			System.exit(-1);
		}
	}

	public String getTrainfileName() {
		return trainfileName;
	}

	public void setTrainfileName(String trainfileName) {
		this.trainfileName = trainfileName;
	}

	public String getTestfileName() {
		return testfileName;
	}

	public void setTestfileName(String testfileName) {
		this.testfileName = testfileName;
	}

	public J48 getTree() {
		return tree;
	}

	public void setTree(J48 tree) {
		this.tree = tree;
	}

	public BufferedReader getTrainreader() {
		return trainreader;
	}

	public void setTrainreader(BufferedReader trainreader) {
		this.trainreader = trainreader;
	}

	public Evaluation getEval() {
		return eval;
	}

	public void setEval(Evaluation eval) {
		this.eval = eval;
	}

	public double getMinRootMeanError() {
		return minRootMeanError;
	}

	public void setMinRootMeanError(double minRootMeanError) {
		this.minRootMeanError = minRootMeanError;
	}

	public int getMaxM() {
		return maxM;
	}

	public void setMaxM(int maxM) {
		this.maxM = maxM;
	}

	public int getMaxN() {
		return maxN;
	}

	public void setMaxN(int maxN) {
		this.maxN = maxN;
	}

	public String getBestCaseSummary() {
		return bestCaseSummary;
	}

	public void setBestCaseSummary(String bestCaseSummary) {
		this.bestCaseSummary = bestCaseSummary;
	}

	public int[] getBestSetting() {
		return bestSetting;
	}

	public void setBestSetting(int[] bestSetting) {
		this.bestSetting = bestSetting;
	}

	public BufferedReader getTestreader() {
		return testreader;
	}

	public void setTestreader(BufferedReader testreader) {
		this.testreader = testreader;
	}

	public Instances getTestingData() {
		return testingData;
	}

	public void setTestingData(Instances testingData) {
		this.testingData = testingData;
	}
}