import type { Grid2Props } from '@mui/material/Unstable_Grid2';
import Grid from '@mui/material/Unstable_Grid2';
import type { ReactNode } from 'react';

interface Props extends Grid2Props {
   children?: ReactNode;
}

export const Column = (props: Props) => {
   const gridProps = { ...props };
   delete gridProps.children;

   return <>
      <Grid { ...gridProps }>
         {props.children}
      </Grid>
   </>
}