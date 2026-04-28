import torch
import torch.nn as nn
from tqdm import tqdm
from kornia.losses import DiceLoss
from typing import Optional, Tuple
import os
import shutil
import pickle

from kornia.losses import DiceLoss


def save_epoch_model(model, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    
    
    base_filename = 'ShadeNet'
    epoch_model_file = f"{base_filename}_ep{epoch + 1}.pt"
    
    # Save model state dict with epoch suffix
    torch.save(model.state_dict(), os.path.join(save_dir, epoch_model_file))
    print(f"Saved epoch {epoch + 1} model: {epoch_model_file}")



def save_final_model_and_metrics(
    model,
    save_dir,
    train_dice_loss,
    train_reconstruction_loss,
    train_bce_loss,
    val_dice_loss,
    val_reconstruction_loss,
    val_bce_loss,
    config_path,
):
    """
    Save a trained ShadeNet model, its configuration, and all training and
    validation loss histories to disk.

    All files are written to ``save_dir``. Directories are created
    automatically if they do not exist.

    Files saved
    -----------
    ``shadenet.pt``
        Model state dict (``model.state_dict()``). Load with:

        .. code-block:: python

            model = ShadeNet(...)
            model.load_state_dict(torch.load('shadenet.pt'))

    ``shadenet_config.pkl``
        The model's ``config_file`` attribute, containing architecture
        hyperparameters (e.g. mid_layers, n_classes). Used to reconstruct
        the model at inference time without hardcoding settings.

    ``train_metrics.pkl``
        Dictionary of perpetual training loss lists:
        ``{"total", "reconstruction", "bce"}``.
        Each value is a list with one entry per epoch (or per step,
        depending on how you populate them).

    ``val_metrics.pkl``
        Dictionary of perpetual validation loss lists:
        ``{"total", "reconstruction", "bce"}``.

    Parameters
    ----------
    model : ShadeNet
        Trained ShadeNet instance. Must have a ``config_file`` attribute
        (set at ``__init__`` time) containing the architecture config.
    save_dir : str
        Path to the output directory. Created automatically if absent.
    train_total_loss : list of float
        Perpetual list of combined training loss values (one per epoch).
    train_reconstruction_loss : list of float
        Perpetual list of reconstruction training loss values.
    train_bce_loss : list of float
        Perpetual list of BCE segmentation training loss values.
    val_total_loss : list of float
        Perpetual list of combined validation loss values.
    val_reconstruction_loss : list of float
        Perpetual list of reconstruction validation loss values.
    val_bce_loss : list of float
        Perpetual list of BCE segmentation validation loss values.

    Notes
    -----
    - All loss lists should be the same length (one entry per epoch).
      No enforcement is done here but mismatched lengths will make
      plotting confusing.
    - Uses ``pickle`` for metrics and config — both are safe to load back
      with ``pickle.load`` in a matching Python/PyTorch environment.
    - ``model.state_dict()`` only saves learned parameters, not the
      architecture. Always save ``config_file`` alongside so you can
      reconstruct the model for inference.

    Example
    -------
    >>> save_final_model_and_metrics(
    ...     model                   = shadenet,
    ...     save_dir                = 'checkpoints/run_01',
    ...     train_total_loss        = train_losses['total'],
    ...     train_reconstruction_loss = train_losses['recon'],
    ...     train_bce_loss          = train_losses['bce'],
    ...     val_total_loss          = val_losses['total'],
    ...     val_reconstruction_loss = val_losses['recon'],
    ...     val_bce_loss            = val_losses['bce'],
    ... )
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── File names ─────────────────────────────────────────────────────────
    MODEL_FILE  = 'shadenet.pt'
    CONFIG_FILE = 'shadenet_config.pkl'
    TRAIN_FILE  = 'train_metrics.pkl'
    VAL_FILE    = 'val_metrics.pkl'

    # ── Model weights ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(save_dir, MODEL_FILE))

    # ── Model config ───────────────────────────────────────────────────────
    with open(os.path.join(save_dir, CONFIG_FILE), 'wb') as f:
        pickle.dump(model.config_file, f)

    # ── Training metrics ───────────────────────────────────────────────────
    train_metrics = {
        'dice':          train_dice_loss,
        'reconstruction': train_reconstruction_loss,
        'bce':            train_bce_loss,
    }
    with open(os.path.join(save_dir, TRAIN_FILE), 'wb') as f:
        pickle.dump(train_metrics, f)

    # ── Validation metrics ─────────────────────────────────────────────────
    val_metrics = {
        'dice':          val_dice_loss,
        'reconstruction': val_reconstruction_loss,
        'bce':            val_bce_loss,
    }
    with open(os.path.join(save_dir, VAL_FILE), 'wb') as f:
        pickle.dump(val_metrics, f)

    shutil.copy(config_path, os.path.join(save_dir, os.path.basename(config_path)))

    # ── Feedback ───────────────────────────────────────────────────────────
    print(f"Saved to: {save_dir}")
    print(f"  Model   : {MODEL_FILE}")
    print(f"  Config  : {CONFIG_FILE}")
    print(f"  Train   : {TRAIN_FILE}")
    print(f"  Val     : {VAL_FILE}")




def train_shadenet(
    model: nn.Module,
    train_loader,
    val_loader,
    save_dir: str = "./models",
    config_path: str = "./",
    n_classes: int = 1,
    device: Optional[torch.device | str] = None,
    EPOCHS: int = 50,
    warmup_epochs: int = 5,
    lr: float = 7e-5,
    ignore_index: int = 255,
    dice_w: float = 0.8,
    mask_w: float = 0.5,
    mse_w: float = 1.0,
    cw: Optional[torch.Tensor] = None,
    accumulation_steps: int = 1,
    
):
    """
    Train ShadeNet for a fixed number of epochs with gradient accumulation
    and mixed precision, logging per-epoch Dice, BCE, and MSE losses for
    both training and validation splits.

    Loss formulation
    ----------------
    Segmentation head (predicted_mask vs mask):
        dice_w * DiceLoss + mask_w * BCEWithLogitsLoss

    Reconstruction head (reconstructed vs target):
        mse_w * MSELoss

    Total:
        loss = dice_w * DiceLoss + mask_w * BCE + mse_w * MSE

    Parameters
    ----------
    model : nn.Module
        ShadeNet instance returning (predicted_mask, reconstructed).
    train_loader : DataLoader
        Training loader. Batches must be dicts with keys:
        "Image", "Mask", "Target", "File Name".
    val_loader : DataLoader
        Validation loader. Same dict schema as train_loader.
    n_classes : int, optional
        Number of segmentation classes. Default: 1.
    device : torch.device or str, optional
        Device to train on. Defaults to CPU if not specified.
    EPOCHS : int, optional
        Number of full passes over the training set. Default: 50.
    lr : float, optional
        Adam learning rate. Default: 7e-5.
    ignore_index : int, optional
        Class index ignored by DiceLoss. Default: 255.
    dice_w : float, optional
        Weight for the Dice loss term. Default: 0.8.
    mask_w : float, optional
        Weight for the BCE loss term. Default: 0.5.
    mse_w : float, optional
        Weight for the MSE reconstruction loss term. Default: 1.0.
    cw : torch.Tensor, optional
        Per-class weights for DiceLoss and BCEWithLogitsLoss to handle
        class imbalance. Shape: (n_classes,). Default: None.
    accumulation_steps : int, optional
        Batches to accumulate before stepping the optimiser.
        Effective batch size = batch_size x accumulation_steps. Default: 1.

    Returns
    -------
    model : nn.Module
        Trained model, still on device.
    train_diceloss : list of float
        Per-epoch mean Dice loss on the training set.
    train_bceloss : list of float
        Per-epoch mean BCE loss on the training set.
    train_mseloss : list of float
        Per-epoch mean MSE loss on the training set.
    val_diceloss : list of float
        Per-epoch mean Dice loss on the validation set.
    val_bceloss : list of float
        Per-epoch mean BCE loss on the validation set.
    val_mseloss : list of float
        Per-epoch mean MSE loss on the validation set.

    Notes
    -----
    - Progress bars update every 300 batches to avoid stdout flooding.
    - A leftover gradient step is performed after the last batch if
      len(train_loader) is not divisible by accumulation_steps.
    - Mixed precision is enabled automatically on CUDA devices.
    """

    # ── Device setup ───────────────────────────────────────────────────────
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device=device)

    # ── Loss functions ─────────────────────────────────────────────────────
    cw = cw.to(device=device, dtype=torch.float32) if cw is not None else None

    dice_criterion = DiceLoss(average='micro', ignore_index=ignore_index, weight=cw)
    bceLoss_criterion = nn.BCEWithLogitsLoss(weight=cw)
    mse_criterion     = nn.MSELoss()

    # ── Optimiser ──────────────────────────────────────────────────────────
    opt = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5)

    # ── Mixed precision ────────────────────────────────────────────────────
    use_amp = (device.type == 'cuda')
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Loss logs (one entry per epoch) ────────────────────────────────────
    train_diceloss, train_bceloss, train_mseloss = [], [], []
    val_diceloss,   val_bceloss,   val_mseloss   = [], [], []


    for epoch in range(EPOCHS):

        # ── Training ───────────────────────────────────────────────────────
        model.train()

        running_loss       = torch.tensor(0.0, device=device)
        diceloss_train_run = torch.tensor(0.0, device=device)
        bceloss_train_run  = torch.tensor(0.0, device=device)
        mseloss_train_run  = torch.tensor(0.0, device=device)

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for batch_idx, data in enumerate(train_loader_tqdm):
            img    = data["Image"].to(device=device,  non_blocking=True)
            mask   = data["Mask"].to(device=device,   non_blocking=True)
            mask = mask.squeeze(1)
            target = data["Target"].to(device=device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                predicted_mask, reconstructed = model(img)

                ramp = min(epoch / warmup_epochs, 1)
                
                lossdice = dice_criterion(predicted_mask, mask)
                lossbce  = bceLoss_criterion(predicted_mask, mask)
                lossmse  = mse_criterion(reconstructed, target)
                loss     = (dice_w * lossdice * ramp + mask_w * lossbce * ramp + mse_w * lossmse) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            running_loss       += loss.detach() * accumulation_steps
            diceloss_train_run += lossdice.detach()
            bceloss_train_run  += lossbce.detach()
            mseloss_train_run  += lossmse.detach()

            if batch_idx % 300 == 0:
                train_loader_tqdm.set_postfix({
                    "Loss": f"{(running_loss / (batch_idx + 1)).item():.4f}",
                    "Dice": f"{(diceloss_train_run / (batch_idx + 1)).item():.4f}",
                    "BCE":  f"{(bceloss_train_run  / (batch_idx + 1)).item():.4f}",
                    "MSE":  f"{(mseloss_train_run  / (batch_idx + 1)).item():.4f}",
                })

        # Flush remaining accumulated gradients
        if len(train_loader) % accumulation_steps != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        # Epoch-level training averages
        epoch_train_loss = (running_loss       / len(train_loader)).item()
        train_diceloss.append((diceloss_train_run / len(train_loader)).item())
        train_bceloss.append((bceloss_train_run   / len(train_loader)).item())
        train_mseloss.append((mseloss_train_run   / len(train_loader)).item())

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()

        val_running     = torch.tensor(0.0, device=device)
        dice_loss_run   = torch.tensor(0.0, device=device)
        bceloss_running = torch.tensor(0.0, device=device)
        mseloss_running = torch.tensor(0.0, device=device)

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")

        with torch.no_grad():
            for i, data in enumerate(val_loader_tqdm):
                img    = data["Image"].to(device=device,  non_blocking=True)
                mask   = data["Mask"].to(device=device,   non_blocking=True)
                target = data["Target"].to(device=device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    predicted_mask, reconstructed = model(img)

                    lossdice = dice_criterion(predicted_mask, mask)
                    lossbce  = bceLoss_criterion(predicted_mask, mask)
                    lossmse  = mse_criterion(reconstructed, target)
                    loss     = dice_w * lossdice + mask_w * lossbce + mse_w * lossmse

                val_running     += loss
                dice_loss_run   += lossdice
                bceloss_running += lossbce
                mseloss_running += lossmse

                if i % 300 == 0:
                    val_loader_tqdm.set_postfix({
                        "Val Loss": f"{(val_running / (i + 1)).item():.4f}",
                    })

            avg_dice = (dice_loss_run   / len(val_loader)).item()
            avg_bce  = (bceloss_running / len(val_loader)).item()
            avg_mse  = (mseloss_running / len(val_loader)).item()

            val_diceloss.append(avg_dice)
            val_bceloss.append(avg_bce)
            val_mseloss.append(avg_mse)

        print(f"Epoch [{epoch+1}/{EPOCHS}]"
              f"\n\t LR              : {lr:.2e}"
              f"\n\t Train Loss      : {epoch_train_loss:.4f}"
              f"\n\t Train Dice Loss : {train_diceloss[-1]:.4f}"
              f"\n\t Train BCE Loss  : {train_bceloss[-1]:.4f}"
              f"\n\t Train MSE Loss  : {train_mseloss[-1]:.4f}"
              f"\n\t Val Dice Loss   : {avg_dice:.4f}"
              f"\n\t Val BCE Loss    : {avg_bce:.4f}"
              f"\n\t Val MSE Loss    : {avg_mse:.4f}")


        if epoch % 3 == 0:
            save_epoch_model(model, save_dir, epoch)

    save_final_model_and_metrics(model, 
                                save_dir, 
                                train_diceloss,
                                train_mseloss,
                                train_bceloss,
                                val_diceloss,
                                val_mseloss,
                                val_bceloss,
                                config_path,
                        )

    return model, train_diceloss, train_mseloss, train_bceloss, val_diceloss, val_mseloss, val_bceloss