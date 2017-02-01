package br.unicamp.ic.execution.knn;

import br.unicamp.ic.classifier.ClassifierExecutor;
import br.unicamp.ic.classifier.KFoldClassifierExecutor;
import br.unicamp.ic.featureselection.AsyncFeatureSelection;
import br.unicamp.ic.featureselection.AsyncFeatureSelection2;
import weka.classifiers.lazy.IBk;

public class ImageSegmentationFeatureSelectionExecution extends FeatureSelectionExecution {

	private AsyncFeatureSelection2 featureSelection;
	
	public ImageSegmentationFeatureSelectionExecution(boolean[] features) {
		super("segment.arff", features, 150, 6, 0.2, new IBk());
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true };
		FeatureSelectionExecution fs = new ImageSegmentationFeatureSelectionExecution(features);
		fs.executeAll();
	}
	
	@Override
	public void executeAll() {
		executeFullFeaturesWithNoFilters();
		executeWithNoFilter();
		executeFullFeaturesNormalized();
		executeWithNormalization();
		executeFullFeaturesZScore();
		executeWithZScore();
	}

	@Override
	public void executeWithNormalization() {
		writeMsg("executeWithNormalization");
		// carrega os atributos e passa os filtros
		featureSelection = new AsyncFeatureSelection2(runtime, limit, mr,
				databaseName, 10, replaceMissingValues, normalize);
		// executa a seleção de atributos
		featureSelection.execute();

	}

	@Override
	public void executeWithZScore() {
		writeMsg("executeWithZScore");
		// carrega os atributos e passa os filtros
		featureSelection = new AsyncFeatureSelection2(runtime, limit, mr,
				databaseName, 10, replaceMissingValues, zscore);
		// executa a seleção de atributos
		featureSelection.execute();
	}

	@Override
	public void executeWithNoFilter() {
		writeMsg("executeWithNoFilter");
		// carrega os atributos e passa os filtros
		featureSelection = new AsyncFeatureSelection2(runtime, limit, mr,
				databaseName, 10, replaceMissingValues);
		// executa a seleção de atributos
		featureSelection.execute();
	}
	
	@Override
	public void executeFullFeaturesWithNoFilters() {
		writeMsg("executeFullFeaturesWithNoFilters");
		ClassifierExecutor executor = new KFoldClassifierExecutor(new IBk());
		executor.loadFeatures(databaseName, replaceMissingValues);
		double result = executor.execute(features, KFOLD);
		writeMsg("Full "+result+" %");

	}	
	@Override
	public void executeFullFeaturesNormalized() {
		writeMsg("executeFullFeaturesNormalized");
		ClassifierExecutor executor = new KFoldClassifierExecutor(new IBk());
		executor.loadFeatures(databaseName, replaceMissingValues, normalize);
		double result = executor.execute(features, KFOLD);
		writeMsg("Full "+result+" %");
		
	}
	
	@Override
	public void executeFullFeaturesZScore() {
		writeMsg("executeFullFeaturesZScore");
		ClassifierExecutor executor = new KFoldClassifierExecutor(new IBk());
		executor.loadFeatures(databaseName, replaceMissingValues, zscore);
		double result = executor.execute(features, KFOLD);
		writeMsg("Full "+result+" %");
		
	}
}
