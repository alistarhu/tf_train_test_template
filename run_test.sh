# example
#python src/main.py --channel_rito 0.5 --num_classes 5 --batch_size 1 --num_test_samples 12097 --is_train 0 --test_model mobilenetv2-400200 --test_record_name expression_dect_test.tfrecord --dataset_dir /media/data1/hl/expression_cls_data/expression_dect_tfrecords64x64
python src/main.py --channel_rito 0.25 --num_classes 5 --batch_size 1 --num_test_samples 31154 --is_train 0 --test_model mobilenetv2-370185 --test_record_name expression_dect_real_test.tfrecord --dataset_dir /media/data1/hl/expression_cls_data/expression_dect_tfrecords64x64
