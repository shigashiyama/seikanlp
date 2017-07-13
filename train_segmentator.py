from datetime import datetime
import sys
import logging
import argparse
import copy
import numpy as np

import chainer
from chainer import serializers
from chainer import cuda

import util
import read_embedding as emb
from conlleval import conlleval


def batch_generator(instances, labels, batchsize, shuffle=True, xp=np):

    len_data = len(instances)
    perm = np.random.permutation(len_data) if shuffle else range(0, len_data)

    for i in range(0, len_data, batchsize):
        i_max = min(i + batchsize, len_data)
        xs = [xp.asarray(instances[perm[i]], dtype=np.int32) for i in range(i, i_max)]
        ts = [xp.asarray(labels[perm[i]], dtype=np.int32) for i in range(i, i_max)]

        yield xs, ts
    raise StopIteration


def evaluate(model, instances, labels, batchsize, xp=np):
    evaluator = model.copy()          # to use different state
    evaluator.predictor.train = False # dropout does nothing
    evaluator.compute_fscore = True

    count = 0
    num_tokens = 0
    total_loss = 0
    total_ecounts = conlleval.EvalCounts()

    for xs, ts in batch_generator(instances, labels, batchsize, shuffle=False, xp=xp):
        with chainer.no_backprop_mode():
            loss, ecounts = evaluator(xs, ts, train=False)

        num_tokens += sum([len(x) for x in xs])
        count += len(xs)
        total_loss += loss.data
        ave_loss = total_loss / num_tokens
        total_ecounts = conlleval.merge_counts(total_ecounts, ecounts)
        c = total_ecounts
        acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
        overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)
           
    print_results(total_loss, ave_loss, count, c, acc, overall)
    stat = 'n_sen: %d, n_token: %d, n_chunk: %d, n_chunk_p: %d' % (count, c.token_counter, c.found_correct, c.found_guessed)
    res = '%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\t%.4f\t%.5f' % ((100.*acc), (100.*overall.prec), (100.*overall.rec), (100.*overall.fscore), overall.tp, overall.fp, overall.fn, total_loss, ave_loss)
            
    return stat, res


def print_results(total_loss, ave_loss, count, c, acc, overall):
    print('total loss: %.4f' % total_loss)
    print('ave loss: %.5f'% ave_loss)
    print('#sen, #token, #chunk, #chunk_pred: %d %d %d %d' %
          (count, c.token_counter, c.found_correct, c.found_guessed))
    print('TP, FP, FN: %d %d %d' % (overall.tp, overall.fp, overall.fn))
    print('A, P, R, F:%6.2f %6.2f %6.2f %6.2f' % 
          (100.*acc, 100.*overall.prec, 100.*overall.rec, 100.*overall.fscore))


def main():

    if chainer.__version__[0] != '2':
        print("chainer version>=2.0.0 is required.")
        sys.exit()

    # get arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--cudnn', dest='use_cudnn', action='store_true')
    parser.add_argument('--eval', action='store_true', help='Evaluate on given model using input date without training')
    parser.add_argument('--limit', type=int, default=-1, help='Limit the number of samples for quick tests')
    parser.add_argument('--batchsize', '-b', type=int, default=20)
    # parser.add_argument('--bproplen', type=int, default=35, help='length of truncated BPTT')
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--resume', default='', help='Resume the training from snapshot')
    parser.add_argument('--resume_epoch', type=int, default=1, help='Resume the training from the epoch')
    parser.add_argument('--iter_to_report', '-i', type=int, default=10000)
    parser.add_argument('--rnn_layer', '-l', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--rnn_unit_type', default='lstm')
    parser.add_argument('--rnn_bidirection', action='store_true')
    parser.add_argument('--crf', action='store_true')
    parser.add_argument('--linear_activation', '-a', default='')
    parser.add_argument('--rnn_hidden_unit', '-u', type=int, default=800)
    parser.add_argument('--embed_dim', '-d', type=int, default=300)
    parser.add_argument('--embed_path', default='')
    parser.add_argument('--lc', type=int, default=0, help='Left context size')
    parser.add_argument('--rc', type=int, default=0, help='Right context size')
    parser.add_argument('--optimizer', '-o', default='sgd')
    parser.add_argument('--lr', '-r', type=float, default=1, help='Value of initial learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0, help='Momentum ratio')
    parser.add_argument('--lrdecay', default='', help='\'start:width:decay rate\'')
    parser.add_argument('--weightdecay', '-w', type=float, default=0, help='Weight decay ratio')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--gradclip', '-c', type=float, default=5,)
    parser.add_argument('--format', '-f', help='Format of input data')
    parser.add_argument('--subpos_depth', '-s', type=int, default=-1)
    parser.add_argument('--dir_path', '-p', help='Directory path of input data (train, valid and test)')
    parser.add_argument('--train_data', '-t', help='Filename of training data')
    parser.add_argument('--validation_data', '-v', help='Filename of validation data')
    parser.add_argument('--test_data', help='Filename of test data')
    parser.add_argument('--tag_schema', default='BIES')
    parser.set_defaults(use_cudnn=False)
    parser.set_defaults(crf=False)
    parser.set_defaults(rnn_bidirection=False)
    args = parser.parse_args()

    # print('# bproplen: {}'.format(args.bproplen))
    print('# No log: {}'.format(args.nolog))
    print('# GPU: {}'.format(args.gpu))
    print('# cudnn: {}'.format(args.use_cudnn))
    print('# eval: {}'.format(args.eval))
    print('# limit: {}'.format(args.limit))
    print('# minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# iteration to report: {}'.format(args.iter_to_report))
    print('# rnn unit type: {}'.format(args.rnn_unit_type))
    print('# rnn bidirection: {}'.format(args.rnn_bidirection))
    print('# crf: {}'.format(args.crf))
    print('# linear_activation: {}'.format(args.linear_activation))
    print('# rnn_layer: {}'.format(args.rnn_layer))
    print('# embedding dimension: {}'.format(args.embed_dim))
    print('# rnn hiddlen unit: {}'.format(args.rnn_hidden_unit))
    print('# pre-trained embedding model: {}'.format(args.embed_path))
    print('# left context size: {}'.format(args.lc))
    print('# right context size: {}'.format(args.rc))
    print('# optimization algorithm: {}'.format(args.optimizer))
    print('# learning rate: {}'.format(args.lr))
    print('# learning rate decay: {}'.format(args.lrdecay))
    if args.lrdecay:
        array = args.lrdecay.split(':')
        lrdecay_start = int(array[0])
        lrdecay_width = int(array[1])
        lrdecay_rate = float(array[2])
        print('# learning rate decay start:', lrdecay_start)
        print('# learning rate decay width:', lrdecay_width)
        print('# learning rate decay rate:',  lrdecay_rate)
    print('# momentum: {}'.format(args.momentum))
    print('# wieght decay: {}'.format(args.weightdecay))
    print('# dropout ratio: {}'.format(args.dropout))
    print('# gradient norm threshold to clip: {}'.format(args.gradclip))
    print('# subpos depth: {}'.format(args.subpos_depth))
    print('# resume: {}'.format(args.resume))
    print('# epoch to resume: {}'.format(args.resume_epoch))
    print('# tag schema: {}'.format(args.tag_schema))
    print(args)
    print('')

    # Prepare logger

    time0 = datetime.now().strftime('%Y%m%d_%H%M')
    print('INFO: start: %s\n' % time0)
    if not args.nolog:
        logger = open('log/' + time0 + '.log', 'a')
        logger.write(str(args)+'\n')

    # Load word embedding model

    embed_model = emb.read_model(args.embed_path) if args.embed_path else None

    # Load dataset

    train_path = args.dir_path + args.train_data
    val_path = args.dir_path + args.validation_data
    test_path = args.dir_path + (args.test_data if args.test_data else '')
    refer_vocab = embed_model.wv if args.embed_path else set()

    limit = args.limit if args.limit > 0 else -1

    train, train_t, token2id, label2id = util.read_data(
        args.format, train_path, subpos_depth=args.subpos_depth,
        schema=args.tag_schema, limit=limit)

    val, val_t, token2id, label2id = util.read_data(
        args.format, val_path, token2id=token2id, label2id=label2id, subpos_depth=args.subpos_depth,
        schema=args.tag_schema, limit=limit)

    if args.test_data:
        test, test_t, token2id, label2id = util.read_data(
            args.format, test_path, token2id=token2id, label2id=label2id, update_token=False,
            refer_vocab=refer_vocab, schema=args.tag_schema, limit=limit)
    else:
        test = []

    n_train = len(train)
    n_val = len(val)
    n_test = len(test)
    n_vocab = len(token2id)
    n_labels = len(label2id)

    t2i_tmp = list(token2id.items())
    id2token = {v:k for k,v in token2id.items()}
    id2label = {v:k for k,v in label2id.items()}
    print('vocab =', n_vocab)
    print('data length: train=%d val=%d test=%d' % (n_train, n_val, n_test))
    print()
    print('train:', train[:3], '...', train[n_train-3:])
    print('train_t:', train_t[:3], '...', train[n_train-3:])
    print()
    print('token2id:', t2i_tmp[:3], '...', t2i_tmp[len(t2i_tmp)-3:])
    print('label2id:', label2id)
    print()
    if not args.nolog:
        logger.write('INFO: vocab = %d\n' % n_vocab)
        logger.write('INFO: data length: train=%d val=%d\n' % (n_train, n_val))

    # gpu settings

    if args.gpu >= 0:
        # Make the specified GPU current
        cuda.get_device_from_id(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np
        
    chainer.config.cudnn_deterministic = args.use_cudnn

    # Set parameters to dictionary

    params = {'embed_dim' : args.embed_dim,
              'rnn_layer' : args.rnn_layer,
              'rnn_hidden_unit' : args.rnn_hidden_unit,
              'rnn_bidirection' : args.rnn_bidirection,
              'crf' : args.crf,
              'linear_activation' : args.linear_activation,
              'left_contexts' : args.lc,
              'right_contexts' : args.rc,
              'dropout' : args.dropout,
              'tag_schema' : args.tag_schema
    }
    if args.embed_path:
        params.update({'embed_path' : args.embed_path})
              
    # Prepare model

    model = util.load_model_from_params(params, model_path=args.resume, token2id=token2id, 
                                        embed_model=embed_model, label2id=label2id, gpu=args.gpu)
    model.compute_fscore = True

    # Set up optimizer

    if args.optimizer == 'sgd':
        if args.momentum <= 0:
            optimizer = chainer.optimizers.SGD(lr=args.lr)
        else:
            optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = chainer.optimizers.Adam()
    elif args.optimizer == 'adagrad':
        optimizer = chainer.optimizers.AdaGrad(lr=args.lr)

    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    if args.weightdecay > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weightdecay))

    # Write parameter file

    if not args.nolog:
        params.update({
            'token2id_path' : 'nn_model/vocab_' + time0 + '.t2i.txt',
            'label2id_path' : 'nn_model/vocab_' + time0 + '.l2i.txt',
            'model_date': time0
        })
        util.write_param_file(params, 'nn_model/param_' + time0 + '.txt')

    # Run

    if args.eval:
        print('<test result>')
        te_stat, te_res = evaluate(model, test, test_t, args.batchsize, xp=xp)
        print()

        time = datetime.now().strftime('%Y%m%d_%H%M')
        if not args.nolog:
            logger.write('finish: %s\n' % time)
            logger.close()
        print('finish: %s\n' % time)
        sys.exit()

    
    n_iter_report = args.iter_to_report
    n_iter = 0
    for e in range(max(1, args.resume_epoch), args.epoch+1):
        time = datetime.now().strftime('%Y%m%d_%H%M')
        if not args.nolog:
            logger.write('INFO: start epoch %d at %s\n' % (e, time))
        print('start epoch %d: %s' % (e, time))

        # learning rate decay

        if args.lrdecay and args.optimizer == 'sgd':
            if (e - lrdecay_start) % lrdecay_width == 0:
                lr_tmp = optimizer.lr
                optimizer.lr = optimizer.lr * lrdecay_rate
                if not args.nolog:
                    logger.write('learning decay: %f -> %f' % (lr_tmp, optimizer.lr))
                print('learning decay: %f -> %f' % (lr_tmp, optimizer.lr))

        count = 0
        total_loss = 0
        num_tokens = 0
        total_ecounts = conlleval.EvalCounts()

        i = 0
        for xs, ts in batch_generator(train, train_t, args.batchsize, shuffle=True, xp=xp):
            loss, ecounts = model(xs, ts, train=True)
            num_tokens += sum([len(x) for x in xs])
            count += len(xs)
            total_loss += loss.data
            total_ecounts = conlleval.merge_counts(total_ecounts, ecounts)
            i_max = min(i + args.batchsize, n_train)
            print('* batch %d-%d loss: %.4f' % ((i+1), i_max, loss.data))
            i = i_max
            n_iter += 1

            optimizer.target.cleargrads() # Clear the parameter gradients
            loss.backward()               # Backprop
            loss.unchain_backward()       # Truncate the graph
            optimizer.update()            # Update the parameters

            # Evaluation
            if (n_iter * args.batchsize) % n_iter_report == 0: # or i == n_train:

                now_e = '%.2f' % (n_iter * args.batchsize / n_train)
                time = datetime.now().strftime('%Y%m%d_%H%M')
                print()
                print('### iteration %s (epoch %s)' % ((n_iter * args.batchsize), now_e))
                print('<training result for previous iterations>')

                ave_loss = total_loss / num_tokens
                c = total_ecounts
                acc = conlleval.calculate_accuracy(c.correct_tags, c.token_counter)
                overall = conlleval.calculate_metrics(c.correct_chunk, c.found_guessed, c.found_correct)
                print_results(total_loss, ave_loss, count, c, acc, overall)
                print()

                if not args.nolog:
                    t_stat = 'n_sen: %d, n_token: %d, n_chunk: %d, n_chunk_p: %d' % (
                        count, c.token_counter, c.found_correct, c.found_guessed)
                    t_res = '%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%d\t%.4f\t%.4f' % (
                        (100.*acc), (100.*overall.prec), (100.*overall.rec), (100.*overall.fscore), 
                        overall.tp, overall.fp, overall.fn, total_loss, ave_loss)
                    logger.write('INFO: train - %s\n' % t_stat)
                    if n_iter == 1:
                        logger.write('data\titer\ep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\ttloss\taloss\n')
                    logger.write('train\t%d\t%s\t%s\n' % (n_iter, now_e, t_res))

                print('<validation result>')
                v_stat, v_res = evaluate(model, val, val_t, args.batchsize, xp=xp)
                print()

                if not args.nolog:
                    logger.write('INFO: valid - %s\n' % v_stat)
                    if n_iter == 1:
                        logger.write('data\titer\ep\tacc\tprec\trec\tfb1\tTP\tFP\tFN\ttloss\taloss\n')
                    logger.write('valid\t%d\t%s\t%s\n' % (n_iter, now_e, v_res))

                # Save the model
                #mdl_path = 'model/rnn_%s_i%.2fk.mdl' % (time0, (1.0 * n_iter / 1000))
                mdl_path = 'nn_model/rnn_%s_i%s.mdl' % (time0, now_e)
                print('save the model: %s\n' % mdl_path)
                if not args.nolog:
                    logger.write('INFO: save the model: %s\n' % mdl_path)
                    serializers.save_npz(mdl_path, model)
        
                # Reset counters
                count = 0
                total_loss = 0
                num_tokens = 0
                total_ecounts = conlleval.EvalCounts()

                if not args.nolog:
                    logger.close()
                    logger = open('log/' + time0 + '.log', 'a')

    time = datetime.now().strftime('%Y%m%d_%H%M')
    if not args.nolog:
        logger.write('finish: %s\n' % time)
        logger.close()
   
if __name__ == '__main__':
    main()
