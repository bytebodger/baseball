import { useEffect, useRef, useState } from 'react';
import type { Viewport } from '../classes/Viewport';
import { defaultViewports } from '../constants/defaultViewports';
import { ViewportId } from '../enums/ViewportId';

export const useViewport = (initialViewports: Viewport[] = defaultViewports) => {
   const getNarrowestWidth = () => window.innerWidth < window.screen.width ? window.innerWidth : window.screen.width;

   const getShortestHeight = () => window.innerHeight < window.screen.height ? window.innerHeight : window.screen.height;

   const getViewportId = (
      width: number = 0,
      viewports: Viewport[] = defaultViewports,
   ): number => {
      const viewport = viewports.find(viewport => width >= viewport.minWidth && width <= viewport.maxWidth);
      return viewport?.id ?? ViewportId.unknown;
   }

   const { md, sm, xs } = ViewportId;
   const [height, setHeight] = useState<number>(getShortestHeight());
   const [id, setId] = useState<number>(getViewportId(
      getNarrowestWidth(),
      initialViewports,
   ));
   const [isMobile, setIsMobile] = useState([md, sm, xs].includes(id));
   const [width, setWidth] = useState<number>(getNarrowestWidth());
   const viewports = useRef<Viewport[]>(initialViewports);

   const resizeWindow = (): void => {
      setHeight(getShortestHeight());
      const id = getViewportId(getNarrowestWidth(), viewports.current);
      setId(id);
      setWidth(getNarrowestWidth());
      setIsMobile([md, sm, xs].includes(id));
   }

   const updateViewports = (newViewports: Viewport[]): void => {
      viewports.current = newViewports;
      setId(getViewportId(width, newViewports));
   }

   useEffect(() => {
      window.addEventListener('resize', resizeWindow);
      return () => window.removeEventListener('resize', resizeWindow);
   }, []);

   return {
      height,
      id,
      isMobile,
      updateViewports,
      viewports: viewports.current,
      width,
   }
}