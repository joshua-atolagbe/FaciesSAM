import numpy as np

def frequency_weighted_IU(eval_segm, gt_segm, split):

    fwius = []
    for i in range(len(eval_segm)):
        pred_mask = eval_segm[i]
        ground_mask = gt_segm[i]

        unique_classes = range(len(pred_mask))
        fIUs = [0]*len(unique_classes)

        for c in unique_classes:

            pred_mask_c = pred_mask[c]
            if c < len(ground_mask):
                true_mask_c = ground_mask[c]

            if (np.sum(pred_mask_c) == 0) or (np.sum(true_mask_c) == 0):
                continue

            n_ii = np.sum(np.logical_and(pred_mask_c, true_mask_c))
            t_i  = np.sum(pred_mask_c)
            n_ij = np.sum(true_mask_c)

            fIU = (t_i * n_ii) / (t_i + n_ij - n_ii)
            fIUs[c] = fIU

        sh = np.array(pred_mask).shape[1] * np.array(pred_mask).shape[2]
        frequency_weighted_IU_ = np.sum(fIUs) / sh

        fwius.append(frequency_weighted_IU_)
    mfwiu = sum(fwius) / len(fwius)
    
    print(f'Frequency Weighted IU: {mfwiu}')

def pixel_accuracy(eval_segm, gt_segm, split):

    upper_ns, middle_ns, lower_ns, rijnland_chalk, scruff, zechstein= [], [], [], [], [], []

    for i in range(len(eval_segm)):
        pred_mask = eval_segm[i]
        ground_mask = gt_segm[i]

        unique_classes = range(len(pred_mask))
        # class_accuracies = [0]*len(unique_classes)
        sum_n_ii = 0
        sum_t_i  = 0
        for c in unique_classes:

            pred_mask_c = pred_mask[c]
            if c < len(ground_mask):
                true_mask_c = ground_mask[c]

            sum_n_ii += np.sum(np.logical_and(pred_mask_c, true_mask_c))
            sum_t_i += np.sum(true_mask_c)

            if (sum_t_i == 0):
                acc = 0
            else:
                acc = sum_n_ii / sum_t_i

            # class_accuracies[c] = acc

            if c == 0:
                upper_ns.append(acc)
            elif c == 1:
                middle_ns.append(acc)
            elif c == 2:
                lower_ns.append(acc)
            elif c == 3:
                rijnland_chalk.append(acc)
            elif c == 4:
                scruff.append(acc)
            elif c == 5:
                zechstein.append(acc)

    upper_acc = sum(upper_ns) / len(upper_ns)
    middle_ns_acc = sum(middle_ns) / len(middle_ns)
    lower_ns_acc = sum(lower_ns) / len(lower_ns)
    rijnland_chalk_acc = sum(rijnland_chalk) / len(rijnland_chalk)
    scruff_acc = sum(scruff) / len(scruff)
    zechstein_acc = sum(zechstein) / len(zechstein)

    mean_iou = (upper_acc + middle_ns_acc + lower_ns_acc + rijnland_chalk_acc + scruff_acc + zechstein_acc) / 6
    print(f'                             Pixel Acc for {split}                   ')
    print('='*80)
    print('class               images          instances             Mask(Pixel Acc)')
    print(f"all                   {len(eval_segm)}               {len(upper_ns)+len(middle_ns)+len(lower_ns)+len(rijnland_chalk)+len(scruff)}                 {mean_iou:.4f}")
    print(f"upper_ns              {len(eval_segm)}               {len(upper_ns)}                 {upper_acc:.4f}")
    print(f"middle_ns             {len(eval_segm)}               {len(middle_ns)}                 {middle_ns_acc:.4f}")
    print(f"lower_ns              {len(eval_segm)}               {len(lower_ns)}                 {lower_ns_acc:.4f}")
    print(f"rijnland_chalk        {len(eval_segm)}               {len(rijnland_chalk)}                 {rijnland_chalk_acc:.4f}")
    print(f"scruff                {len(eval_segm)}               {len(scruff)}                 {scruff_acc:.4f}")
    print(f"zechstein             {len(eval_segm)}               {len(zechstein)}                 {zechstein_acc:.4f}")


def class_accuracy(eval_segm, gt_segm, split):

    upper_ns, middle_ns, lower_ns, rijnland_chalk, scruff, zechstein= [], [], [], [], [], []

    for i in range(len(eval_segm)):
        pred_mask = eval_segm[i]
        ground_mask = gt_segm[i]

        unique_classes = range(len(pred_mask))
        # class_accuracies = [0]*len(unique_classes)

        for c in unique_classes:

            pred_mask_c = pred_mask[c]
            if c < len(ground_mask):
                true_mask_c = ground_mask[c]

            correct_pixels = np.sum(pred_mask_c & true_mask_c)
            total_pixels = np.sum(true_mask_c)

            if total_pixels != 0:
                acc = correct_pixels / total_pixels
            else:
                acc = 0.0

            # class_accuracies[c] = acc

            if c == 0:
                upper_ns.append(acc)
            elif c == 1:
                middle_ns.append(acc)
            elif c == 2:
                lower_ns.append(acc)
            elif c == 3:
                rijnland_chalk.append(acc)
            elif c == 4:
                scruff.append(acc)
            elif c == 5:
                zechstein.append(acc)

    upper_acc = sum(upper_ns) / len(upper_ns)
    middle_ns_acc = sum(middle_ns) / len(middle_ns)
    lower_ns_acc = sum(lower_ns) / len(lower_ns)
    rijnland_chalk_acc = sum(rijnland_chalk) / len(rijnland_chalk)
    scruff_acc = sum(scruff) / len(scruff)
    zechstein_acc = sum(zechstein) / len(zechstein)

    mean_iou = (upper_acc + middle_ns_acc + lower_ns_acc + rijnland_chalk_acc + scruff_acc + zechstein_acc) / 6
    print(f'                             Class Acc for {split}                   ')
    print('='*80)
    print('class               images          instances             Mask(Class Acc)')
    print(f"all                   {len(eval_segm)}               {len(upper_ns)+len(middle_ns)+len(lower_ns)+len(rijnland_chalk)+len(scruff)}                 {mean_iou:.4f}")
    print(f"upper_ns              {len(eval_segm)}               {len(upper_ns)}                 {upper_acc:.4f}")
    print(f"middle_ns             {len(eval_segm)}               {len(middle_ns)}                 {middle_ns_acc:.4f}")
    print(f"lower_ns              {len(eval_segm)}               {len(lower_ns)}                 {lower_ns_acc:.4f}")
    print(f"rijnland_chalk        {len(eval_segm)}               {len(rijnland_chalk)}                 {rijnland_chalk_acc:.4f}")
    print(f"scruff                {len(eval_segm)}               {len(scruff)}                 {scruff_acc:.4f}")
    print(f"zechstein             {len(eval_segm)}               {len(zechstein)}                 {zechstein_acc:.4f}")


def mIoU(eval_segm, gt_segm, split):

    upper_ns, middle_ns, lower_ns, rijnland_chalk, scruff, zechstein= [], [], [], [], [], []

    for i in range(len(eval_segm)):
        pred_mask = eval_segm[i]
        ground_mask = gt_segm[i]

        unique_classes = range(len(pred_mask))
        # IUs = [0]*len(unique_classes)

        for c in unique_classes:

            pred_mask_c = pred_mask[c]
            if c < len(ground_mask):
                true_mask_c = ground_mask[c]

            if (np.sum(pred_mask_c) == 0) or (np.sum(true_mask_c) == 0):
                continue

            n_ii = np.sum(np.logical_and(pred_mask_c, true_mask_c))
            t_i  = np.sum(pred_mask_c)
            n_ij = np.sum(true_mask_c)

            IU = n_ii / (t_i + n_ij - n_ii)
            # IU[i] = IU

            if c == 0:
                upper_ns.append(IU)
            elif c == 1:
                middle_ns.append(IU)
            elif c == 2:
                lower_ns.append(IU)
            elif c == 3:
                rijnland_chalk.append(IU)
            elif c == 4:
                scruff.append(IU)
            elif c == 5:
                zechstein.append(IU)

    upper_IU = sum(upper_ns) / len(upper_ns)
    middle_ns_IU = sum(middle_ns) / len(middle_ns)
    lower_ns_IU = sum(lower_ns) / len(lower_ns)
    rijnland_chalk_IU = sum(rijnland_chalk) / len(rijnland_chalk)
    scruff_IU = sum(scruff) / len(scruff)
    zechstein_IU = sum(zechstein) / len(zechstein)

    mean_iou = (upper_IU + middle_ns_IU + lower_ns_IU + rijnland_chalk_IU + scruff_IU + zechstein_IU) / 6
    print(f'                             mIoU for `{split}` split                   ')
    print('='*80)
    print('class               images          instances             Mask(mIoU)')
    print(f"all                   {len(eval_segm)}               {len(upper_ns)+len(middle_ns)+len(lower_ns)+len(rijnland_chalk)+len(scruff)}                 {mean_iou:.4f}")
    print(f"upper_ns              {len(eval_segm)}               {len(upper_ns)}                 {upper_IU:.4f}")
    print(f"middle_ns             {len(eval_segm)}               {len(middle_ns)}                 {middle_ns_IU:.4f}")
    print(f"lower_ns              {len(eval_segm)}               {len(lower_ns)}                 {lower_ns_IU:.4f}")
    print(f"rijnland_chalk        {len(eval_segm)}               {len(rijnland_chalk)}                 {rijnland_chalk_IU:.4f}")
    print(f"scruff                {len(eval_segm)}               {len(scruff)}                 {scruff_IU:.4f}")
    print(f"zechstein             {len(eval_segm)}               {len(zechstein)}                 {zechstein_IU:.4f}")

