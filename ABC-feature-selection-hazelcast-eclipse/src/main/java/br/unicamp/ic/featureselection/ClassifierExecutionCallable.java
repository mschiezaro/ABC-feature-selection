package br.unicamp.ic.featureselection;

import java.io.Serializable;
import java.util.concurrent.Callable;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.filters.Filter;
import br.unicamp.ic.classifier.ClassifierExecutor;
import br.unicamp.ic.classifier.KFoldClassifierExecutor;
import br.unicamp.ic.featureselection.swarm.FoodSource;

/**
 * Usado para paralelizar a execução de classificação
 * 
 */
public class ClassifierExecutionCallable implements Callable<FoodSource>,
		Serializable {

	private static final long serialVersionUID = 8378169067678190547L;

	private boolean[] features;

	private Instances originalInstances;
	
	private Classifier classifier;

	/**
	 * Estratégia de execução do classificador
	 */
	private static final int KFOLD = 10;

	public ClassifierExecutionCallable(boolean[] features,
			Instances originalInstances, Classifier classifier,
			Filter... filters) {
		
		this.features = features;
		this.originalInstances = originalInstances;
		this.classifier = classifier;

	}

	@Override
	public FoodSource call() throws Exception {
		FoodSource f = new FoodSource(features, calculateFitness(features), 1);
		return f;
	}

	/**
	 * Calcula a qualidade de uma fonte de alimento através da fonte selecionada
	 * 
	 * @param features
	 *            Features utilizadas para o cálculo da qualidade da fonte de
	 *            alimento
	 * @return valor de 0 a 100 que indica a qualidade da fonte
	 */
	private double calculateFitness(boolean features[]) {

		ClassifierExecutor executor = new KFoldClassifierExecutor(classifier);
		// carrega as features
		executor.setOriginalInstances(originalInstances);
		executor.loadFeatures();
		// chama o classificador e rtorna a acurácia
		return executor.execute(features, KFOLD);
	}

}