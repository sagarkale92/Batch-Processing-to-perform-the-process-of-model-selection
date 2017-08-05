//package batchProcessing.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesSimple;
import weka.core.Instances;

public class NaiveBayesSmp implements LearningAlgorithm {
	private String trainfileName;
	private String testfileName;
	private NaiveBayesSimple model = null;
	private BufferedReader trainreader = null;
	private BufferedReader testreader = null;
	private Instances trainingData = null;
	private Instances testingData = null;
	private Evaluation eval = null;
	private double minRootMeanError;
	private String bestCaseSummary;

	/**
	 * Initializes the parameter.
	 */
	@Override
	public void initialize(String trainfileNameIn, String testfileNameIn) {
		try {
			System.out.format(
					"Naive Bayes Simple\n------------------------------------------------------------------------------------------------------------------------------------------------");
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
			setModel(new NaiveBayesSimple());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(-1);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	@Override
	public void initialize(String s1, String s2, int m, int n) {
	}

	/**
	 * Processes the data using naive bayes simple algorithm.
	 */
	@Override
	public void process() {
		try {
			model.buildClassifier(trainingData);
			Evaluation eval = new Evaluation(trainingData);
			Random rand = new Random(1);
			int folds = 10;
			eval.crossValidateModel(model, trainingData, folds, rand);
			this.setMinRootMeanError(eval.meanAbsoluteError());
			this.setBestCaseSummary(eval.toSummaryString());
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}
	
	/**
	 * Processes the test data using naive bayes simple algorithm.
	 */
	@Override
	public void processTest() {
		try{
			setModel(new NaiveBayesSimple());
			model.buildClassifier(testingData);
			Evaluation eval = new Evaluation(testingData);
			eval.evaluateModel(model, testingData);
			System.out.println(eval.toSummaryString());
			for (int i = 0; i < testingData.numInstances(); i++) {
				double pred = model.classifyInstance(testingData.instance(i));
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

	public NaiveBayesSimple getModel() {
		return model;
	}

	public void setModel(NaiveBayesSimple model) {
		this.model = model;
	}

	public BufferedReader getTrainreader() {
		return trainreader;
	}

	public void setTrainreader(BufferedReader trainreader) {
		this.trainreader = trainreader;
	}

	public Instances getTrainingData() {
		return trainingData;
	}

	public void setTrainingData(Instances trainingData) {
		this.trainingData = trainingData;
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

	public String getBestCaseSummary() {
		return bestCaseSummary;
	}

	public void setBestCaseSummary(String bestCaseSummary) {
		this.bestCaseSummary = bestCaseSummary;
	}

	public Instances getTestingData() {
		return testingData;
	}

	public void setTestingData(Instances testingData) {
		this.testingData = testingData;
	}

	public BufferedReader getTestreader() {
		return testreader;
	}

	public void setTestreader(BufferedReader testreader) {
		this.testreader = testreader;
	}
}