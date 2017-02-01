package br.unicamp.ic.hazelcast;

import com.hazelcast.core.Hazelcast;

public class HazelCastMachine {

	public static void main(String[] args) {
		int nrOfNodes = 0;
		if (args.length == 0) {
			nrOfNodes = 1;
		} else {
			nrOfNodes = Integer.parseInt(args[0]);
		}
		for (int i = 0; i < nrOfNodes; i++) {
			Hazelcast.newHazelcastInstance();
		}
	}
}
