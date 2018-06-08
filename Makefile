# ----------------------------------------------------------------
# environment
CC		= mpiicc
FC		= 

# ----------------------------------------------------------------
# options

#CFLAGS		= -Ofast -xCORE-AVX2 -qopt-report
CFLAGS		= -O3 -xMIC-AVX512 -qopt-report
FFLAGS		= 

# ----------------------------------------------------------------
# sources and objects

C_SRC		= mat-mat.c
F_SRC		= 

C_OBJ		= $(C_SRC:.c=)
F_OBJ		= $(F_SRC:.f=)

# ----------------------------------------------------------------
# executables

EXEC		= $(C_OBJ) 

all:		$(EXEC)

$(C_OBJ):	$(C_SRC)
	$(CC) -o $@ $(CFLAGS) $(C_SRC) -lm


# ----------------------------------------------------------------
# rules

.c.:
	$(CC) -o $* $(CFLAGS) -c $<

.f.:
	$(FC) -o $* $(FFLAGS) -c $<

# ----------------------------------------------------------------
# clean up

clean:
	/bin/rm -f $(EXEC) $(F_SRC:.f=.o)

# ----------------------------------------------------------------
# End of Makefile
