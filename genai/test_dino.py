import argparse
import os
import copy
import cv2
import torch
import matplotlib.pyplot as plt
import dino_sam_inpainting as D

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=False, default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py', help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=False, default='./models/groundingdino_swint_ogc.pth', help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, default='./models/sam_vit_h_4b8939.pth', help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=False, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cuda", help="running on gpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = D.load_image(image_path)
    # load model
    model = D.load_model(config_file, grounded_checkpoint, device=device)
    
    output_file_name = f'{format(os.path.basename(image_path))}'

    # visualize raw image
    image_pil.save(os.path.join(output_dir, output_file_name))

    # run grounding dino model
    boxes_filt, pred_phrases = D.get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    if use_sam_hq:
        predictor = D.SamPredictor(D.build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = D.SamPredictor(D.build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        D.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        D.show_box(box.numpy(), plt.gca(), label)

    output_file_name = f'{format(os.path.basename(image_path))}'
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f'grounded_sam_{output_file_name}'),
        bbox_inches="tight", dpi=200, pad_inches=0.0
    )

    D.save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
