obj_model=scratch/model 
obj_latent=scratch/latent.png

all: $(obj_model) $(obj_latent) 

$(obj_model): 
	python VAE.py train

$(obj_latent): $(obj_model)
	python VAE.py test
	
board:
	tensorboard --logdir=logs --host=0.0.0.0

clean: 
	rm -rf logs $(obj_model) $(obj_latent)
