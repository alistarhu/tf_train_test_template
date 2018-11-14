import tensorflow as tf
import net_def
import time
import glob
import os
import tfrecord_write_read as tfwr
import argparse
import numpy as np

def get_paramters():
    parser = argparse.ArgumentParser(description="MobileNetV2")
    parser.add_argument('--model_name', type=str, default='mobilenetv2')
    parser.add_argument('--num_samples', type=int, help='the number of train samples')
    parser.add_argument('--num_valid_samples', type=int, help='the number of test samples')
    parser.add_argument('--num_test_samples', type=int, help='the number of test samples')
    
    parser.add_argument('--epoch', type=int, default=500)######
    parser.add_argument('--batch_size', type=int, default=64)#####
    parser.add_argument('--num_classes', type=int, default=5)#####
    parser.add_argument('--height', type=int, default=64)#####
    parser.add_argument('--width', type=int, default=64)######
    parser.add_argument('--channel', type=int, default=3)
    
    parser.add_argument('--learning_rate', type=float, default=0.01)#0.001
    parser.add_argument('--lr_decay', type=float, default=0.8685)#0.98
    parser.add_argument('--num_epochs_per_decay', type=int, default=4)#10
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.9)# for adam
    parser.add_argument('--momentum', type=float, default=0.9)# for momentum
    
    parser.add_argument('--save_model_per_num_epoch', type=int, default=10)
    parser.add_argument('--channel_rito',type=float, default=1)
    parser.add_argument('--dataset_dir', type=str, default='./tfrecords', help='tfrecord file dir')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--test_record_name', type=str, default='expression_dect_test.tfrecord')
   
    parser.add_argument('--test_model', type=str)
    #parser.add_argument('--gpu', dest='gpu' ,action='store_false')

    parser.add_argument('--renew', type=bool, default=False)
    parser.add_argument('--is_train', type=int)

    args = parser.parse_args()
    return args

def load(sess, saver, checkpoint_dir):
    import re
    print("[*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("[*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print("[*] Failed to find a checkpoint")
        return False, 0


def main():
    # Read parameters setting
    args = get_paramters()

    # Set the CUDA id. if not set the id, sys will use the all available card
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" # use one card
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0,2" # use multi-card
    #os.environ['CUDA_VISIBLE_DEVICES'] = "-1" # not use gpu

    #print('the channel rito is: {}\n'.format(args.channel_rito))
    sess = tf.Session()
    # Train the model
    if args.is_train == 1:
        # build graph
        inputs = tf.placeholder(tf.float32, [None, args.height, args.width, 3], name='input')
        labels = tf.placeholder(tf.int32, name='label')
        logits, pred = net_def.mobilenetv2_addBias(inputs, num_classes=args.num_classes, channel_rito=args.channel_rito, is_train=True)# with channel rito channel_rito

        # loss
        loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))   # L2 regularization
        total_loss = loss_ + l2_loss

        # evaluate model, for classification
        correct_pred = tf.equal(tf.cast(tf.argmax(pred, 1), tf.int32), labels)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # saver for save/restore model
        saver = tf.train.Saver(max_to_keep=None)#save checkpoint each epoch

        # load data
        glob_pattern_train = os.path.join(args.dataset_dir, 'expression_dect_train.tfrecord')
        glob_pattern_valid = os.path.join(args.dataset_dir, 'expression_dect_valid.tfrecord')
        img_batch, label_batch = tfwr.get_batch(glob_pattern_train, args.batch_size, args.height, args.width, args.channel, shuffle=True)
        img_batch_valid, label_batch_valid = tfwr.get_batch(glob_pattern_valid, args.batch_size, args.height, args.width, args.channel)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        # Write the train result 
        f = open('train_log.txt','w')
        
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)

        # learning rate decay
        base_lr = tf.constant(args.learning_rate)
        num_batches_per_epoch = args.num_samples // args.batch_size
        lr_decay_step = int(num_batches_per_epoch * args.num_epochs_per_decay)
        
        #global_step = tf.placeholder(dtype=tf.float32, shape=())
        global_step = tf.Variable(0)
        lr = tf.train.exponential_decay(base_lr, global_step=global_step, decay_steps=lr_decay_step, decay_rate=args.lr_decay, staircase=True)
        # optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.9, momentum=0.9)
            if args.optimizer == 'adam':
                train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=args.beta1).minimize(total_loss, global_step=global_step)
                #print('use adam!!!\n')
            elif args.optimizer == 'momentum':
                train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=args.momentum).minimize(total_loss, global_step=global_step)
                #print('use mom!!!\n')
            else:
                train_op = tf.train.GradientDescentOptimizer(lr)
        
        # summary writer
        if not os.path.exists(args.logs_dir):
            os.mkdir(args.logs_dir)
            os.makedirs(os.path.join(args.logs_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(args.logs_dir, 'validate'), exist_ok=True)
        writer_train = tf.summary.FileWriter(os.path.join(args.logs_dir, 'train'), sess.graph)
        writer_validate = tf.summary.FileWriter(os.path.join(args.logs_dir, 'validate'))
        
        # summary
        loss_summary = tf.summary.scalar('total_loss', total_loss)
        acc_summary = tf.summary.scalar('accuracy', acc)
        lr_summary = tf.summary.scalar('learning_rate', lr)
        merge_summary_train = tf.summary.merge([loss_summary, acc_summary, lr_summary])
        merge_summary_valid = tf.summary.merge([loss_summary, acc_summary])
        #summary_op = tf.summary.merge_all()
        
        # variable init
        sess.run(tf.global_variables_initializer())
        step = 0
        # start from a point
        if args.renew:
            print('[*] Try to load trained model...')
            could_load, step = load(sess, saver, args.checkpoint_dir)

        max_steps = int(args.num_samples / args.batch_size * args.epoch)

        print('START TRAINING...')
        for _step in range(step+1, max_steps+1):
            start_time = time.time()
            train_img_batch, train_label_batch = sess.run([img_batch, label_batch])
            
            #feed_dict = {global_step:_step, inputs:train_img_batch, labels:train_label_batch}
            global_step = _step
            feed_dict = {inputs:train_img_batch, labels:train_label_batch}

            # train each step
            _, _lr, summary_train, _loss, _acc = sess.run([train_op, lr, merge_summary_train, total_loss, acc], feed_dict=feed_dict)
            writer_train.add_summary(summary_train, _step)
            # print logs and write summary per epoch
            if _step % num_batches_per_epoch == 0:
                # run validate per each epoch
                valid_loss_sum = 0
                valid_acc_sum = 0
                valid_step = args.num_valid_samples//args.batch_size
                for i in range(valid_step):
                    valid_img_batch, valid_label_batch = sess.run([img_batch_valid, label_batch_valid])
                    valid_dict = {inputs: valid_img_batch, labels: valid_label_batch}
                    valid_loss, valid_acc = sess.run([total_loss, acc], feed_dict=valid_dict)
                    valid_loss_sum += valid_loss
                    valid_acc_sum += valid_acc
                valid_loss_sum = valid_loss_sum/valid_step
                valid_acc_sum = valid_acc_sum/valid_step
                
                # summary test
                summary_valid = sess.run(merge_summary_valid, feed_dict={total_loss:valid_loss_sum, acc:valid_acc_sum})
                writer_validate.add_summary(summary_valid, _step)
            
                print('epoch:{0}  global_step:{1}, time:{2:.3f}, lr:{3:.8f}, train acc:{4:.6f}, train loss:{5:.6f}, valid acc:{6:.6f}, \
valid loss:{7:.6f}'.format(_step//num_batches_per_epoch, _step, time.time() - start_time, _lr, _acc, _loss, valid_acc_sum, valid_loss_sum))
                f.write('epoch:{0}  global_step:{1}, time:{2:.3f}, lr:{3:.8f}, train acc:{4:.6f}, train loss:{5:.6f}, valid acc:{6:.6f}, \
valid loss:{7:.6f}\n'.format(_step//num_batches_per_epoch, _step, time.time() - start_time, _lr, _acc, _loss, valid_acc_sum, valid_loss_sum))
            # save model each 10 epoch
            if _step % (num_batches_per_epoch*args.save_model_per_num_epoch) == 0:
                save_path = saver.save(sess, os.path.join(args.checkpoint_dir, args.model_name), global_step=_step)
                print('Current model saved in ' + save_path)
                f.write('Current model saved in {}\n'.format(save_path))

        tf.train.write_graph(sess.graph_def, args.checkpoint_dir, args.model_name + '.pb')
        save_path = saver.save(sess, os.path.join(args.checkpoint_dir, args.model_name), global_step=max_steps)
        print('Final model saved in ' + save_path)
        f.write('Final model saved in {}\n'.format(save_path))
        f.close()
        coord.request_stop()
        coord.join(threads)
        print 'FINISHED TRAINING.'
        
    # Test the model
    else:
        class_count = np.zeros((args.num_classes, args.num_classes), dtype=np.int32)
        # build graph
        inputs = tf.placeholder(tf.float32, [None, args.height, args.width, 3], name='input')
        labels = tf.placeholder(tf.int32, name='label')
        logits, pred = net_def.mobilenetv2_addBias(inputs, num_classes=args.num_classes, channel_rito=args.channel_rito, is_train=False) # with channel rito channel_rito

        # evaluate model, for classification
        pred_class = tf.cast(tf.argmax(pred, 1), tf.int32)
        correct_pred = tf.equal(tf.cast(tf.argmax(pred, 1), tf.int32), labels)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # saver for save/restore model
        saver = tf.train.Saver()
        
        # load pretrained model
        # load model for Test
        checkpoint_dir = './checkpoints'
        file_name = args.test_model
        saver.restore(sess,os.path.join(checkpoint_dir, file_name))
        
        f = open('test_log.txt','w')
        # load test data
        glob_pattern = os.path.join(args.dataset_dir, args.test_record_name)

        img_batch, label_batch = tfwr.get_batch(glob_pattern, args.batch_size, args.height, args.width, args.channel, shuffle=False)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        batch_num = args.num_test_samples//args.batch_size
        total_acc = 0
        for ii in range(batch_num):
            test_img_batch, test_label_batch = sess.run([img_batch, label_batch])
            feed_dict={inputs:test_img_batch, labels:test_label_batch}
            predict_class, test_acc=sess.run([pred_class, acc], feed_dict=feed_dict)
            total_acc = total_acc + test_acc
            for jj in range(args.batch_size):
                class_count[test_label_batch[jj], predict_class[jj]] +=1
        np.savetxt('test_log.txt', class_count, fmt='%s',newline='\n')
        f = open('test_log.txt','a')
        f.write('\n')
        for i in range(args.num_classes):
            f.write('class {} acc is {:.4f}\n'.format(i, class_count[i][i]*1.0/np.sum(class_count, axis=1)[i]))
        #f.close()

        print("The total test acc is :{:.6f}".format(total_acc*1.0/batch_num))
        f.write(os.path.join(checkpoint_dir, file_name)+'\n')
        f.write("The total test acc is :{:.6f}\n".format(total_acc*1.0/batch_num))
        f.close()
        coord.request_stop()
        coord.join(threads)
        print 'FINISHED TESTING.'

    sess.close()


if __name__=='__main__':
    main()
