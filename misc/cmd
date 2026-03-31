# Region-based with SLIC (best, needs scikit-image)
python unified_ig_v2.py --region-insdel --viz-region-insdel

# Grid fallback (no extra deps)
python  uigv3.py --region-insdel --no-slic --patch-size 14

# Both pixel and region side-by-side
python  uigv3.py --insdel --region-insdel --viz-insdel --viz-region-insdel

# Custom grid size (larger patches = fewer regions = coarser test)
python uigv3.py --region-insdel --no-slic --patch-size 40 --viz --viz-path my_heatmaps.png
python uigv4.py --region-insdel --no-slic --patch-size 40 --viz --viz-path my_heatmaps.png
python uigv4o.py --region-insdel --no-slic --patch-size 40 --viz --viz-path my_heatmaps.png

Source: ./sample_imagenet1k/n03187595_dial_telephone.JPEG   seed 6                                            
