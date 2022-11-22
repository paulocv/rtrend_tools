
# --- Inputs
PROJECT_DIR=.   # Use ../.. to compile from the source directory
SRC_DIR="$PROJECT_DIR"/rtrend_tools/rt_mcmc
SOURCES_REL="Rt.c"
GSL_FLAGS="-lgsl -lgslcblas"
COMPILER=gcc

# --- Create full paths (still relative to project_dir)
for NAME in $SOURCES_REL
do
    SOURCES_ABS=$SOURCES_ABS" "$SRC_DIR"/"$NAME
done

# --- Compilation command
$COMPILER -g -Wall $GSL_FLAGS $SOURCES_ABS -o "$PROJECT_DIR"/main_mcmc_rt
