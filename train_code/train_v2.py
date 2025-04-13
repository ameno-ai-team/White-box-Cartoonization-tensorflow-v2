import tensorflow as tf
import utils
import os
import numpy as np
import argparse
from network_v2 import UnetGenerator, Discriminator
from tqdm import tqdm
from guided_filter import guided_filter

# Force CPU usage
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)     
    parser.add_argument("--total_iter", default=100000, type=int)
    parser.add_argument("--adv_train_lr", default=2e-4, type=float)
    parser.add_argument("--gpu_fraction", default=0.5, type=float)
    parser.add_argument("--save_dir", default='train_cartoon')
    parser.add_argument("--use_enhance", default=False)
    parser.add_argument("--continue_from", default=None, type=str,
                        help="Path to checkpoint directory to continue training from")
    return parser.parse_args()

class CartoonGAN:
    def __init__(self, args):
        self.args = args
        
        # Initialize models
        self.generator = UnetGenerator()
        self.discriminator = Discriminator()
        
        # Build models with dummy inputs
        dummy_input = tf.random.normal([1, args.patch_size, args.patch_size, 3])
        _ = self.generator(dummy_input)
        _ = self.discriminator(dummy_input)
        
        # Initialize optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(args.adv_train_lr, beta_1=0.5, beta_2=0.99)
        self.d_optimizer = tf.keras.optimizers.Adam(args.adv_train_lr, beta_1=0.5, beta_2=0.99)
        
        # Initialize optimizer states
        self.g_optimizer.build(self.generator.trainable_variables)
        self.d_optimizer.build(self.discriminator.trainable_variables)
        
        # Create checkpoint manager
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, args.save_dir + '/ckpts', max_to_keep=3
        )
        
        # Try to restore checkpoint
        if args.continue_from:
            try:
                status = self.checkpoint.restore(tf.train.latest_checkpoint(args.continue_from))
                # Don't assert exact match since we're loading from TF1.x checkpoint
                status.expect_partial()
                print(f"Restored from {args.continue_from}")
            except Exception as e:
                print(f"Error restoring checkpoint: {e}")
                print("Starting training from scratch...")
        else:
            print("Starting training from scratch...")
    
    @tf.function
    def train_step(self, photo_batch, cartoon_batch):
        # Train discriminator first
        with tf.GradientTape() as tape:
            gen_output = self.generator(photo_batch, training=True)
            disc_real = self.discriminator(cartoon_batch, training=True)
            disc_fake = self.discriminator(gen_output, training=True)
            
            # Calculate superpixel
            superpixel_batch = utils.simple_superpixel(gen_output, seg_num=200)
            disc_smooth = self.discriminator(superpixel_batch, training=True)
            
            # Discriminator losses
            d_loss_real = tf.reduce_mean(tf.square(disc_real - 1.0))
            d_loss_fake = tf.reduce_mean(tf.square(disc_fake))
            d_loss_smooth = tf.reduce_mean(tf.square(disc_smooth))
            d_loss = d_loss_real + d_loss_fake + 0.1 * d_loss_smooth
            
        # Apply discriminator gradients
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        
        # Train generator
        with tf.GradientTape() as tape:
            gen_output = self.generator(photo_batch, training=True)
            disc_fake = self.discriminator(gen_output, training=True)
            
            # Calculate superpixel
            superpixel_batch = utils.simple_superpixel(gen_output, seg_num=200)
            disc_smooth = self.discriminator(superpixel_batch, training=True)
            
            # Generator losses
            g_loss_fake = tf.reduce_mean(tf.square(disc_fake - 1.0))
            g_loss_smooth = tf.reduce_mean(tf.square(disc_smooth - 1.0))
            g_loss = g_loss_fake + 0.1 * g_loss_smooth
            
        # Apply generator gradients
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        
        return g_loss, d_loss
    
    def generator_loss(self, disc_fake, gen_output, real_image):
        adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_fake), logits=disc_fake
        ))
        l1_loss = tf.reduce_mean(tf.abs(gen_output - real_image))
        total_loss = adv_loss + 100 * l1_loss
        return total_loss
    
    def discriminator_loss(self, disc_real, disc_fake):
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_real), logits=disc_real
        ))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(disc_fake), logits=disc_fake
        ))
        return real_loss + fake_loss
    
    def train(self):
        # Load datasets
        face_photo_list = utils.load_image_list('dataset/photo_face')
        face_cartoon_list = utils.load_image_list('dataset/cartoon_face')
        
        print(f"Found {len(face_photo_list)} photo images and {len(face_cartoon_list)} cartoon images")
        
        if len(face_photo_list) == 0 or len(face_cartoon_list) == 0:
            raise ValueError("No images found in dataset directories!")
        
        for iteration in tqdm(range(self.args.total_iter)):
            # Get next batch of face images
            photo_batch = utils.next_batch(face_photo_list, self.args.batch_size)
            cartoon_batch = utils.next_batch(face_cartoon_list, self.args.batch_size)
            
            # Convert to tensors
            photo_batch = tf.convert_to_tensor(photo_batch, dtype=tf.float32)
            cartoon_batch = tf.convert_to_tensor(cartoon_batch, dtype=tf.float32)
            
            # Training step
            g_loss, d_loss = self.train_step(photo_batch, cartoon_batch)
            
            # Save checkpoint periodically
            if (iteration + 1) % 500 == 0:
                self.manager.save()
                print(f'Iteration {iteration + 1}, G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}')

def main():
    args = arg_parser()
    model = CartoonGAN(args)
    model.train()

if __name__ == '__main__':
    main()
