NN: main.o NeuralNet.o
	g++ -std=c++11 -g -Wall main.o NeuralNet.o -o NN

main.o: main.cpp NeuralNet.h
	g++ -std=c++11 -g -Wall -c main.cpp

NeuralNet.o: NeuralNet.h NeuralNet.cpp
	g++ -std=c++11 -g -Wall -c NeuralNet.cpp

clean:
	rm -rf NN *.o
