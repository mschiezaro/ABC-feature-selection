package br.unicamp.ic.execution.knn;

import br.unicamp.ic.featureselection.AsyncFeatureSelection2;
import br.unicamp.ic.util.FileUtil;
import weka.classifiers.lazy.IBk;

public class ArrhythmiaYale64x64FeatureSelectionExecution extends
		FeatureSelectionExecution {
	
	private AsyncFeatureSelection2 featureSelection;


	public ArrhythmiaYale64x64FeatureSelectionExecution(boolean[] features) {
		super("arrhythmia.arff", features, 20, 30, 0.2, new IBk());
	}
	
	public static void main(String[] args) {
		boolean features[] = { true, true, true, true, true, true, true, true,true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true,
				true, true, true, true, true, true, true, true, true, true};

		FeatureSelectionExecution fs = new ArrhythmiaYale64x64FeatureSelectionExecution(
				features);
		fs.executeWithNoFilter();
		FileUtil.newInstance().close();
	}
	
	public void executeWithNoFilter() {
		writeMsg("executeWithNoFilter");
		// carrega os atributos e passa os filtros
		featureSelection = new AsyncFeatureSelection2(runtime, limit, mr,databaseName, 8, replaceMissingValues);
		// executa a seleção de atributos
		featureSelection.execute();
	}
}
