nn: nn.c
	gcc -std=c99 -O0 nn.c -o nn
    
clean:
	rm nn
