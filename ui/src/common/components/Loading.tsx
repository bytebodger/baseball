import { CircularProgress, Dialog } from '@mui/material';

interface Props {
   open: boolean,
}

export const Loading = (props: Props) => <>
   <Dialog
      keepMounted={true}
      open={props.open}
   >
      <div
         style={{
            alignItems: 'center',
            display: 'flex',
            height: '10vh',
            justifyContent: 'center',
            overflow: 'hidden',
            width: '10vw',
         }}
      >
         <CircularProgress variant={'indeterminate'} />
      </div>
   </Dialog>
</>