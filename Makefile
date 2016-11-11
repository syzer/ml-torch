nn: nn.c
	gcc -std=c99 -O3 nn.c -o nn -lm
    
clean:
	rm nn
