# Guide: Resize Windows Partition to Give Linux More Space

## Current Disk Layout

```
Total Disk: 931.5GB NVMe SSD
├─ nvme0n1p1: 100MB   (EFI Boot)
├─ nvme0n1p2: 16MB    (Windows Reserved)
├─ nvme0n1p3: 673.4GB (Windows 11 - NTFS) ⚠️ Can shrink this
├─ nvme0n1p4: 257.3GB (Linux / - ext4)    ✓ Your main partition (98% full!)
└─ nvme0n1p5: 711MB   (Windows Recovery)
```

**Problem:** Linux partition is 98% full (235GB/257GB used)  
**Solution:** Shrink Windows from 673GB to give Linux more space

## Recommended Approach

### Option 1: Conservative (Safest)
- Shrink Windows to **400GB** (still plenty for Windows)
- Give Linux **~270GB more** (total ~527GB)
- Final: Windows 400GB, Linux 527GB

### Option 2: Aggressive (If you rarely use Windows)
- Shrink Windows to **200GB** (minimum comfortable size)
- Give Linux **~470GB more** (total ~727GB)
- Final: Windows 200GB, Linux 727GB

### Option 3: Minimal Windows (If Windows is emergency-only)
- Shrink Windows to **100GB** (bare minimum)
- Give Linux **~570GB more** (total ~827GB)
- Final: Windows 100GB, Linux 827GB

## Step-by-Step Process

### ⚠️ IMPORTANT: Backup First!
Before resizing, backup important data from both partitions!

### Method 1: Using GParted (Recommended - Safest)

#### 1. Create GParted Live USB
```bash
# Download GParted Live ISO
# https://gparted.org/download.php

# Create bootable USB (from Linux)
sudo dd if=gparted-live-*.iso of=/dev/sdX bs=4M status=progress
# Replace /dev/sdX with your USB drive (check with lsblk)
```

#### 2. Boot into GParted
1. Restart computer
2. Boot from GParted USB
3. Select default options to start GParted

#### 3. Resize Partitions
1. **Shrink Windows (nvme0n1p3):**
   - Right-click on nvme0n1p3 (673.4GB NTFS)
   - Select "Resize/Move"
   - Drag the right edge left to shrink
   - Leave desired size (e.g., 200GB for Windows)
   - Click "Resize/Move"

2. **Expand Linux (nvme0n1p4):**
   - Right-click on nvme0n1p4 (257.3GB ext4)
   - Select "Resize/Move"
   - Drag the left edge left to use freed space
   - Click "Resize/Move"

3. **Apply Changes:**
   - Click the green checkmark (Apply All Operations)
   - Wait for completion (may take 30-60 minutes)
   - Reboot when done

### Method 2: From Windows (Less Safe)

#### 1. Shrink Windows Partition
1. Boot into Windows 11
2. Press Win+X, select "Disk Management"
3. Right-click Windows (C:) partition
4. Select "Shrink Volume"
5. Enter amount to shrink (in MB)
   - For 200GB Windows: shrink by ~473,000 MB
   - For 400GB Windows: shrink by ~273,000 MB
6. Click "Shrink"

#### 2. Expand Linux Partition
1. Reboot into Linux
2. Install GParted: `sudo apt install gparted`
3. Run: `sudo gparted`
4. Select nvme0n1p4 (your Linux partition)
5. Resize to use unallocated space
6. Apply changes
7. Reboot

### Method 3: From Linux (Advanced)

⚠️ **WARNING:** This requires unmounting your root partition. Use GParted Live USB instead.

## Post-Resize Verification

After resizing, verify everything worked:

```bash
# Check partition sizes
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE

# Check filesystem
df -h /

# Check for errors
sudo fsck /dev/nvme0n1p4
```

## Recommended: My Suggestion for You

Based on your usage (Linux for AI/ML, Windows rarely):

**Resize to:**
- Windows: **200GB** (plenty for occasional use)
- Linux: **~727GB** (tons of space for models, Docker, experiments)

This gives you:
- 3x more space for Linux
- Enough Windows space for updates and occasional use
- Room for multiple LLM models and Docker images

## Safety Checklist

Before you start:
- [ ] Backup important Windows files
- [ ] Backup important Linux files
- [ ] Ensure laptop is plugged in (don't resize on battery!)
- [ ] Close all applications
- [ ] Disable Windows Fast Startup
- [ ] Run Windows disk check: `chkdsk C: /f`
- [ ] Have 2-3 hours available for the process

## Disable Windows Fast Startup (Important!)

Fast Startup can cause issues with partition resizing:

1. Boot into Windows
2. Control Panel → Power Options
3. "Choose what the power buttons do"
4. "Change settings that are currently unavailable"
5. Uncheck "Turn on fast startup"
6. Save changes
7. Restart Windows normally
8. Then proceed with resizing

## If Something Goes Wrong

If you encounter issues:
1. Boot into GParted Live USB
2. Use "Check" option to verify filesystems
3. Use "Repair" if needed
4. Don't panic - partitions can usually be recovered

## Alternative: Add External Storage

If resizing feels risky, consider:
- External SSD (1TB ~$80-100)
- Mount as `/data` for models and Docker volumes
- Keep system partitions as-is

## Need Help?

The process is generally safe but can be intimidating. Consider:
1. Practice on a VM first
2. Ask for help on Linux forums
3. Have a backup plan (external drive ready)

---

**My Recommendation:** Use GParted Live USB method. It's the safest and most reliable approach.

Good luck! 🚀
