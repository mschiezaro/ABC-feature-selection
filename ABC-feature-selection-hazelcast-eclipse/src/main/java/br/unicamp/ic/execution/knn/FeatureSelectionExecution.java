package br.unicamp.ic.execution.knn;

import java.io.BufferedWriter;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;
import br.unicamp.ic.classifier.ClassifierExecutor;
import br.unicamp.ic.classifier.KFoldClassifierExecutor;
import br.unicamp.ic.featureselection.FeatureSelection;
import br.unicamp.ic.util.FileUtil;

public class FeatureSelectionExecution {

	// Divisões para treinamento e teste
	protected int KFOLD = 10;

	// arquivo com os dados
	protected String databaseName;

	// Filtro para ignorar os values que não existem para
	// determinado atributo
	protected ReplaceMissingValues replaceMissingValues;

	// Filtro z-score
	protected Standardize zscore;

	// Normaliza atributos de 0 a 1
	protected Normalize normalize;

	private FeatureSelection featureSelection;

	protected boolean[] features;
	
	protected BufferedWriter writer;
	
	protected int runtime;
	
	protected int limit;
	
	protected double mr;
	
	private FileUtil fileUtil;

	public FeatureSelectionExecution(String databaseName, boolean[] features,
			int runtime, int limit, double mr, Classifier classifier) {
		this.databaseName = databaseName;
		this.features = features;
		replaceMissingValues = new ReplaceMissingValues();
		zscore = new Standardize();
		normalize = new Normalize();
		fileUtil = FileUtil.newInstance(); 
		writer = fileUtil.getWriter();		
		this.runtime = runtime;
		this.limit = limit;
		this.mr = mr;
	}
	
	public void executeAll() {
		executeFullFeaturesWithNoFilters();		
		executeWithNoFilter();
		executeFullFeaturesNormalized();		
		executeWithNormalization();
		executeFullFeaturesZScore();		
		executeWithZScore();
	}

	public void executeWithNoFilter() {
		writeMsg("executeWithNoFilter");
		featureSelection = new FeatureSelection(runtime, limit, mr, databaseName, replaceMissingValues);
		// executa a seleção de atributos
		featureSelection.execute();
	}
	
	public void executeWithNormalization() {
		writeMsg("executeWithNormalization");
		featureSelection = new FeatureSelection(runtime, limit, mr, databaseName, replaceMissingValues, normalize);
		// executa a seleção de atributos
		featureSelection.execute();

	}

	public void executeWithZScore() {
		writeMsg("executeWithZScore");
		featureSelection = new FeatureSelection(runtime, limit, mr, databaseName, replaceMissingValues, zscore);
		// executa a seleção de atributos
		featureSelection.execute();
	}
	
	public void executeFullFeaturesWithNoFilters() {
		writeMsg("executeFullFeaturesWithNoFilters");
		ClassifierExecutor executor = new KFoldClassifierExecutor(new IBk());		
		executor.loadFeatures(databaseName, replaceMissingValues);
		double result = executor.execute(features, KFOLD);
		writeMsg("Full "+result+" %");
		
	}

	public void executeFullFeaturesNormalized() {
		writeMsg("executeFullFeaturesNormalized");		
		ClassifierExecutor executor = new KFoldClassifierExecutor(new IBk());		
		executor.loadFeatures(databaseName, replaceMissingValues, normalize);
		double result = executor.execute(features, KFOLD);
		writeMsg("Full "+result+" %");		
	}

	public void executeFullFeaturesZScore() {
		writeMsg("executeFullFeaturesZScore");
		ClassifierExecutor executor = new KFoldClassifierExecutor(new IBk());
		executor.loadFeatures(databaseName, replaceMissingValues, zscore);
		double result = executor.execute(features, KFOLD);
		writeMsg("Full "+result+" %");		
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
			throw new RuntimeException();
		}
	}
	
	public FileUtil getFileUtil() {
		return fileUtil;
	}
}

