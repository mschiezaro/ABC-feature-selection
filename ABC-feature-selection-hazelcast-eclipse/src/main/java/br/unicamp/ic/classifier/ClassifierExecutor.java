package br.unicamp.ic.classifier;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;
import weka.filters.Filter;

/**
 * Define a excução de um classificador
 * 
 * @author Mauricio Schiezaro
 * 
 */
public abstract class ClassifierExecutor {
	
	/**
	 * Dados que serão carregados dos arquivos
	 */
	protected Instances originalInstances;

	/**
	 * Dados que são copiados de originalInstances, para podemos manipular quais
	 * atributos iremos utlizar em cada execução.
	 */
	protected Instances instances;

	/**
	 * Carrega do arquivo os dados que serão classificados. Esse método é
	 * chamado uma única vez para carregarmos os dados do arquivo, depois para
	 * recuperar os dados originais é só chamar o método loadFeatures() sem
	 * parâmetros que os dados serão recuperados da memória.
	 * 
	 * @param filename
	 *            Nome do arquivo que será carregado
	 * @param Filtros
	 *            no pré-processamento da feature. Exemplos: normalização,
	 *            Z-score, Replace Missing values
	 */
	public int loadFeatures(String filename, Filter... filter) {
		try {
			// carrega os dados do arquivo
			originalInstances = new Instances(new FileReader(
					System.getProperty("user.dir") + "/src/main/resources/"
							+ filename));
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		if (filter != null) {
			for (int i = 0; i < filter.length; i++) {

				try {
					filter[i].setInputFormat(originalInstances);
				} catch (Exception e) {
					throw new RuntimeException(e);
				}
				try {
					originalInstances = Filter.useFilter(originalInstances,
							filter[i]);
				} catch (Exception e) {
					throw new RuntimeException(e);
				}

			}
		}
		// criamos sempre uma cópia dos dados originais para manipularmos os
		// atributos
		instances = new Instances(originalInstances);
		// retorna o número de atributos carregados do arquivo
		return originalInstances.numAttributes() - 1;		
	}
	
	/**
	 * Carrega do arquivo os dados que serão classificados. Esse método é
	 * chamado uma única vez para carregarmos os dados do arquivo, depois para
	 * recuperar os dados originais é só chamar o método loadFeatures() sem
	 * parâmetros que os dados serão recuperados da memória.
	 * 
	 * @param filename
	 *            Nome do arquivo que será carregado
	 */
	public int loadFeatures(String filename) {
		Filter[] filter =  new Filter[0];
		return loadFeatures(filename, filter);
	}
	
	/**
	 * Carrega os dados originais da mémoria.
	 */
	public void loadFeatures() {
		instances = new Instances(originalInstances);
	}
	
	/**
	 * Retorna o número features dos dados originais carregados do arquivo
	 */
	public int getFeatureSize() {
		if (originalInstances == null) {
			return -1;
		}
		return originalInstances.numAttributes() - 1;
	}
	
	public void setOriginalInstances(Instances originalInstances) {
		this.originalInstances = originalInstances;
	}

	/**
	 * Executa a classificação. Os dados de treinamento e teste utilizam k-fold
	 * onde o número de k é indicado pelo parâmetro kfold, e as features que
	 * serão utilizadas na classificação são indicadas pelo vetor
	 * featureInclusion. Cada posição do vetor indica uma feature e o valor true
	 * indica que ela participará da classificação, false não participará
	 * @param featureInclusion  features que serão utilizadas na classificação
	 * @param k Parâmetro k-fold para dividir dados de treinamento e teste
	 */
	public abstract double execute(boolean[] featureInclusion, int k);
	
	/**
	 * Executa a classificação. Os dados de treinamento e teste utilizam k-fold
	 * onde o número de k é indicado pelo parâmetro kfold, e as features que
	 * serão utilizadas na classificação são indicadas pelo vetor
	 * featureInclusion. Cada posição do vetor indica uma feature e o valor true
	 * indica que ela participará da classificação, false não participará
	 * @param featureInclusion  features que serão utilizadas na classificação
	 * @param k Parâmetro k-fold para dividir dados de treinamento e teste
	 * @param classIndex indica qual coluna está o label da classificação (ground truth)
	 */	
	public abstract double execute(boolean[] featureInclusion, int kFold, int classIndex);
}
