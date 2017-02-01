package br.unicamp.ic.classifier;

import java.io.BufferedWriter;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;
import br.unicamp.ic.util.FileUtil;

public class ClassificationExecutorForComparation {

	// Divisões para treinamento e teste
	private int KFOLD = 10;

	// arquivo com os dados
	protected String databaseName;

	// Filtro para ignorar os values que não existem para
	// determinado atributo
	private ReplaceMissingValues replaceMissingValues;

	// Filtro z-score
	private Standardize zscore;

	// Classificador
	private Classifier classifier;

	// Executa a classificação
	private ClassifierExecutor executor;

	private boolean[] features;

	protected BufferedWriter writer;

	public ClassificationExecutorForComparation(String databaseName,
			boolean[] features, String filename) {

		this.databaseName = databaseName;
		this.features = features;
		replaceMissingValues = new ReplaceMissingValues();
		zscore = new Standardize();
		classifier = new LibSVM();
		executor = new KFoldClassifierExecutor(classifier);
		writer = FileUtil.newInstance(filename).getWriter();
	}

	public void executeAll() {
		executeWithNoFilter();
		executeWithZScore();
	}

	public void executeWithNoFilter() {
		writeMsg("executeWithNoFilter");
		// carrega os atributos e passa os filtros
		executor.loadFeatures(databaseName, replaceMissingValues);
		double result = executor.execute(features, KFOLD);
		writeMsg("No Filter " + result + " %");
	}

	public void executeWithZScore() {
		writeMsg("executeWithZScore");
		// carrega os atributos e passa os filtros
		executor.loadFeatures(databaseName, replaceMissingValues, zscore);
		double result = executor.execute(features, KFOLD);
		writeMsg("Z score " + result + " %");
	}

	public void setDatabaseName(String databaseName) {
		this.databaseName = databaseName;
	}

	protected void writeMsg(String msg) {
		try {
			writer.write(msg);
			writer.newLine();
			writer.flush();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	protected void closeFile() {
		try {
			writer.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public static void main(String[] args) {

		boolean features[] = { true, false, false, true, true, false, false,
				false, true, false, false, true, false, false, false, false,
				false, false, true, false, true, true, false, false, false };
		ClassificationExecutorForComparation executor = new ClassificationExecutorForComparation(
				"autos.arff", features, "AutosPso.log");
		executor.writeMsg("Autos PSO");
		executor.executeAll();
		executor.closeFile();
	
		features = new boolean[] { true, true, true, true, true, false, false,
				false, true, false, true, true, false, false, false, false,
				false, false, true, false, true, false, false, false, false };
		executor = new ClassificationExecutorForComparation("autos.arff",
				features, "AutosAnt1.log");
		executor.writeMsg("Autos Ant");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, false, false, true, true, false,
				false, false, true, false, false, true, false, false, false,
				false, true, false, true, false, false, false, true, true,
				false };
		executor = new ClassificationExecutorForComparation("autos.arff",
				features, "AutosGA.log");
		executor.writeMsg("Autos GA");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, true, true, false, true, true, true,
				true, true };
		executor = new ClassificationExecutorForComparation(
				"breast-cancer.arff", features, "Breast-cancerPsoGA.log");
		executor.writeMsg("breast-cancer PSO and GA");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, true, true, true, true, true, true,
				true, true };
		executor = new ClassificationExecutorForComparation(
				"breast-cancer.arff", features, "Breast-cancerAnt.log");
		executor.writeMsg("breast-cancer Ant");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, true, true, true, true, true, true,
				true };
		executor = new ClassificationExecutorForComparation("diabetes.arff",
				features, "DiabetesPsoAntGA.log");
		executor.writeMsg("diabetes PSO, Ant and GA");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, false, true, false, true, false, true,
				true, false, true, false, true, true };
		executor = new ClassificationExecutorForComparation(
				"heart-statlog.arff", features, "Heart-stalogPso.log");
		executor.writeMsg("heart-statlog PSO");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, false, true, true, true, false, false,
				true, false, true, false, true, false };
		executor = new ClassificationExecutorForComparation(
				"heart-statlog.arff", features, "Heart-stalogAnt.log");
		executor.writeMsg("heart-statlog Ant");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, true, true, true, true, true, true,
				true, false, false, false, false, false };
		executor = new ClassificationExecutorForComparation(
				"heart-statlog.arff", features, "Heart-stalogGA.log");
		executor.writeMsg("heart-statlog GA");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, false, false, false, false, false,
				false, false, true, false, true, false, false, true, true,
				false, true, true, false };
		executor = new ClassificationExecutorForComparation("hepatitis.arff",
				features, "HepatitisPso.log");
		executor.writeMsg("hepatitis PSO");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, false, false, false, true, false,
				false, true, false, false, false, false, false, true, true,
				true, true, false, false };
		executor = new ClassificationExecutorForComparation("hepatitis.arff",
				features, "HepatitisAnt.log");
		executor.writeMsg("hepatitis Ant");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, false, false, false, false, true,
				false, false, false, false, true, false, false, true, true,
				true, true, false, false };
		executor = new ClassificationExecutorForComparation("hepatitis.arff",
				features, "HepatitisGA.log");
		executor.writeMsg("hepatitis GA");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, true, true, true };
		executor = new ClassificationExecutorForComparation("iris.arff",
				features, "IrisPsoAntGA.log");
		executor.writeMsg("iris PSO, Ant and GA");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { false, true, false, false, false, true,
				false, false, true, true, false, true, false, false, false,
				false };
		executor = new ClassificationExecutorForComparation("labor.arff",
				features, "LaborPso.log");
		executor.writeMsg("labor PSO");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { false, true, true, false, false, true,
				false, false, false, false, true, false, false, true, false,
				true };
		executor = new ClassificationExecutorForComparation("labor.arff",
				features, "LaborAnt.log");
		executor.writeMsg("labor Ant");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, false, true, false, false, true, true,
				false, false, false, true, true, true, false, true, true };
		executor = new ClassificationExecutorForComparation("labor.arff",
				features, "LaborGA.log");
		executor.writeMsg("labor GA");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, true, true, true, true, true, true,
				false, true };
		executor = new ClassificationExecutorForComparation("glass.arff",
				features, "GlassPsoAntGA.log");
		executor.writeMsg("glass PSO, Ant and GA");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, true, false, false, false, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true };
		executor = new ClassificationExecutorForComparation("segment.arff",
				features, "SegmentPSOAnt.log");
		executor.writeMsg("Segment PSO and Ant");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, true, false, true, true, true, true,
				true, false, true, true, true, true, true, true, true, true,
				true, true };
		executor = new ClassificationExecutorForComparation("segment.arff",
				features, "SegmentGA.log");
		executor.writeMsg("Segment GA");
		executor.executeAll();
		executor.closeFile();
		
		features = new boolean[] { true, false, true, false, true, false, true,
				true, false, true, false, true, true};
		executor = new ClassificationExecutorForComparation("heart-c.arff",
				features, "HertCPSO.log");
		executor.writeMsg("Hert-C PSO");
		executor.executeAll();
		executor.closeFile();
		
		features = new boolean[] { true, false, true, true, true, false, false,
				true, false, true, false, true, false};
		executor = new ClassificationExecutorForComparation("heart-c.arff",
				features, "HertCAnt.log");
		executor.writeMsg("Hert-C Ant");
		executor.executeAll();
		executor.closeFile();

		features = new boolean[] { true, false, true, true, true, false, false,
				true, false, false, true, true, false};
		executor = new ClassificationExecutorForComparation("heart-c.arff",
				features, "HertCGA.log");
		executor.writeMsg("Hert-C GA");
		executor.executeAll();
		executor.closeFile();

	}
}
