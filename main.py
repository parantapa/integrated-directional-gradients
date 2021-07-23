from IDG.utils import load_model
from transformers import XLNetTokenizer, BertTokenizer
import torch
from IDG.calculate_gradients import execute_IDG
import click


@click.command()
@click.option("-n", "--model_name", required=True, type=str,
              help="name of the model options bert, xlnet")
@click.option("-mp", "--model_path", required=True, type=click.Path(),
              help="Path to the model file.")
@click.option("-i", "--input_file_path", required=True, type=click.Path(),
              help="Path to input file")
@click.option("-cls", "--class_gt", required=True, type=str,
              help="Ground-truth class 0(negative) or 1(positive)")
@click.option("-o", "--output_path", required=True, type=click.Path(),
              help="Path of the output file.", default='')
def main(model_name, model_path, input_file_path, class_gt, output_path):
    model = load_model(model_name, model_path, device=torch.device("cpu"))
    bert = None
    if model_name == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        bert = False
    elif model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        bert = True
    else:
        tokenizer = None
    cls = int(class_gt)
    trees = []
    with open(input_file_path) as fs:
        for line in fs:
            string = line.strip()
            trees.append(string)
    execute_IDG(trees, model, tokenizer, cls, output_path, bert)


if __name__ == '__main__':
    main()