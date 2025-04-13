'''
Source code for CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'
by Xinrui Wang and Jinze yu
'''


import tensorflow as tf
import tensorflow_addons as tfa

import utils
import os
import numpy as np
import argparse
import network 
from tqdm import tqdm
from guided_filter import guided_filter

# Enable memory growth to avoid memory allocation issues
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Enable mixed precision for better performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default = 256, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)     
    parser.add_argument("--total_iter", default = 100000, type = int)
    parser.add_argument("--adv_train_lr", default = 2e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--save_dir", default = 'train_cartoon', type = str)
    parser.add_argument("--use_enhance", default = False)
    parser.add_argument("--continue_from", default = None, type = str, 
                        help = "Path to checkpoint directory to continue training from")

    args = parser.parse_args()
    
    return args



def train(args):
    

    input_photo = tf.keras.Input(shape=[args.patch_size, args.patch_size, 3], batch_size=args.batch_size)
    input_superpixel = tf.keras.Input(shape=[args.patch_size, args.patch_size, 3], batch_size=args.batch_size)
    input_cartoon = tf.keras.Input(shape=[args.patch_size, args.patch_size, 3], batch_size=args.batch_size)
    
    output = network.unet_generator(input_photo)
    output = guided_filter(input_photo, output, r=1)

    
    blur_fake = guided_filter(output, output, r=5, eps=2e-1)
    blur_cartoon = guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

    gray_fake, gray_cartoon = utils.color_shift(output, input_cartoon)
    
    d_loss_gray, g_loss_gray = loss.lsgan_loss(network.disc_sn, gray_cartoon, gray_fake, 
                                             scale=1, patch=True, name='disc_gray')
    d_loss_blur, g_loss_blur = loss.lsgan_loss(network.disc_sn, blur_cartoon, blur_fake, 
                                             scale=1, patch=True, name='disc_blur')


    vgg_model = loss.Vgg19('vgg19_no_fc.npy')
    vgg_photo = vgg_model.build_conv4_4(input_photo)
    vgg_output = vgg_model.build_conv4_4(output)
    vgg_superpixel = vgg_model.build_conv4_4(input_superpixel)
    h, w, c = vgg_photo.get_shape().as_list()[1:]
    
    photo_loss = tf.reduce_mean(tf.losses.absolute_difference(vgg_photo, vgg_output))/(h*w*c)
    superpixel_loss = tf.reduce_mean(tf.losses.absolute_difference\
                                     (vgg_superpixel, vgg_output))/(h*w*c)
    recon_loss = photo_loss + superpixel_loss
    tv_loss = loss.total_variation_loss(output)
    
    g_loss_total = 1e4*tv_loss + 1e-1*g_loss_blur + g_loss_gray + 2e2*recon_loss
    d_loss_total = d_loss_blur + d_loss_gray

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'gene' in var.name]
    disc_vars = [var for var in all_vars if 'disc' in var.name] 
    
    
    tf.summary.scalar('tv_loss', tv_loss)
    tf.summary.scalar('photo_loss', photo_loss)
    tf.summary.scalar('superpixel_loss', superpixel_loss)
    tf.summary.scalar('recon_loss', recon_loss)
    tf.summary.scalar('d_loss_gray', d_loss_gray)
    tf.summary.scalar('g_loss_gray', g_loss_gray)
    tf.summary.scalar('d_loss_blur', d_loss_blur)
    tf.summary.scalar('g_loss_blur', g_loss_blur)
    tf.summary.scalar('d_loss_total', d_loss_total)
    tf.summary.scalar('g_loss_total', g_loss_total)
      
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        
        g_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)\
                                        .minimize(g_loss_total, var_list=gene_vars)
        
        d_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)\
                                        .minimize(d_loss_total, var_list=disc_vars)
           
    '''
    # Using tf.function for better performance
    @tf.function
    def train_step(photo_batch, cartoon_batch, superpixel_batch):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # Generator forward pass
            inter_out = generator(photo_batch, training=True)
            
            # Calculate losses
            g_loss = g_loss_total(photo_batch, superpixel_batch, cartoon_batch, inter_out)
            d_loss = d_loss_total(photo_batch, superpixel_batch, cartoon_batch, inter_out)
            
        # Calculate gradients
        g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
        
        # Apply gradients
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        
        return g_loss, d_loss
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    
    train_writer = tf.summary.FileWriter(args.save_dir+'/train_log')
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(var_list=gene_vars, max_to_keep=20)
   
    with tf.device('/device:GPU:0'):

        sess.run(tf.global_variables_initializer())
        
        # Load checkpoint based on continue_from argument
        if args.continue_from:
            checkpoint_path = args.continue_from
            print(f'Continuing training from checkpoint: {checkpoint_path}')
        else:
            checkpoint_path = 'pretrain/saved_models'
            print('Starting new training from pretrained model')
            
        checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if checkpoint:
            print(f'Loading checkpoint: {checkpoint}')
            saver.restore(sess, checkpoint)
        else:
            print(f'No checkpoint found in {checkpoint_path}')

        face_photo_dir = 'dataset/photo_face'
        face_photo_list = utils.load_image_list(face_photo_dir)
        scenery_photo_dir = 'dataset/photo_scenery'
        scenery_photo_list = utils.load_image_list(scenery_photo_dir)

        face_cartoon_dir = 'dataset/cartoon_face'
        face_cartoon_list = utils.load_image_list(face_cartoon_dir)
        scenery_cartoon_dir = 'dataset/cartoon_scenery'
        scenery_cartoon_list = utils.load_image_list(scenery_cartoon_dir)

        for total_iter in tqdm(range(args.total_iter)):

            if np.mod(total_iter, 5) == 0: 
                photo_batch = utils.next_batch(face_photo_list, args.batch_size)
                cartoon_batch = utils.next_batch(face_cartoon_list, args.batch_size)
            else:
                photo_batch = utils.next_batch(scenery_photo_list, args.batch_size)
                cartoon_batch = utils.next_batch(scenery_cartoon_list, args.batch_size)
        
            inter_out = sess.run(output, feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: photo_batch,
                                                    input_cartoon: cartoon_batch})

            '''
            adaptive coloring has to be applied with the clip_by_value 
            in the last layer of generator network, which is not very stable.
            to stabiliy reproduce our results, please use power=1.0
            and comment the clip_by_value function in the network.py first
            If this works, then try to use adaptive color with clip_by_value.
            '''
            if args.use_enhance:
                superpixel_batch = utils.selective_adacolor(inter_out, power=1.2)
            else:
                superpixel_batch = utils.simple_superpixel(inter_out, seg_num=200)
                
            _, g_loss, r_loss = sess.run([g_optim, g_loss_total, recon_loss],  
                                            feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: superpixel_batch,
                                                    input_cartoon: cartoon_batch})

            _, d_loss, train_info = sess.run([d_optim, d_loss_total, summary_op],  
                                            feed_dict={input_photo: photo_batch, 
                                                    input_superpixel: superpixel_batch,
                                                    input_cartoon: cartoon_batch})


            train_writer.add_summary(train_info, total_iter)
            
            if np.mod(total_iter+1, 50) == 0:

                print('Iter: {}, d_loss: {}, g_loss: {}, recon_loss: {}'.\
                        format(total_iter, d_loss, g_loss, r_loss))
                if np.mod(total_iter+1, 500 ) == 0:
                    saver.save(sess, args.save_dir+'/saved_models/model', 
                               write_meta_graph=False, global_step=total_iter)
                     
                    photo_face = utils.next_batch(face_photo_list, args.batch_size)
                    cartoon_face = utils.next_batch(face_cartoon_list, args.batch_size)
                    photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)
                    cartoon_scenery = utils.next_batch(scenery_cartoon_list, args.batch_size)

                    result_face = sess.run(output, feed_dict={input_photo: photo_face, 
                                                            input_superpixel: photo_face,
                                                            input_cartoon: cartoon_face})

                    result_scenery = sess.run(output, feed_dict={input_photo: photo_scenery, 
                                                                input_superpixel: photo_scenery,
                                                                input_cartoon: cartoon_scenery})

                    utils.write_batch_image(result_face, args.save_dir+'/images', 
                                            str(total_iter)+'_face_result.jpg', 4)
                    utils.write_batch_image(photo_face, args.save_dir+'/images', 
                                            str(total_iter)+'_face_photo.jpg', 4)

                    utils.write_batch_image(result_scenery, args.save_dir+'/images', 
                                            str(total_iter)+'_scenery_result.jpg', 4)
                    utils.write_batch_image(photo_scenery, args.save_dir+'/images', 
                                            str(total_iter)+'_scenery_photo.jpg', 4)

            
if __name__ == '__main__':
    
    args = arg_parser()
    train(args)  
   