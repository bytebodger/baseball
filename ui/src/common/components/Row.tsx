import type { Grid2Props } from '@mui/material/Unstable_Grid2';
import Grid from '@mui/material/Unstable_Grid2';
import type { ReactNode } from 'react';

interface Props extends Grid2Props {
   children?: ReactNode;
}

export const Row = (props: Props) => {
   const gridProps = {
      columnSpacing: {
         xs: 1,
         sm: 2,
         md: 3,
      },
      rowSpacing: {
         xs: 1,
         sm: 2,
         md: 3,
      },
      ...props,
   };
   delete gridProps.children;

   return <>
      <Grid
         container={true}
         { ...gridProps }
      >
         {props.children}
      </Grid>
   </>
}